from abc import ABC, abstractmethod
import os

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from gomoku.core.gomoku import GameState, Gomoku
from gomoku.inference.base import InferenceClient
from gomoku.utils.config.loader import MctsConfig

from ..treenode import TreeNode


def _safe_dirichlet_noise(
    policy: torch.Tensor, legal_mask: torch.Tensor, epsilon: float, alpha: float
) -> torch.Tensor:
    """
    Apply Dirichlet noise on legal actions and renormalize to a probability distribution.

    Notes
    -----
    Samples ``noise ~ Dirichlet(alpha)`` over the legal actions, then mixes it with
    the original policy:

    ``mixed = (1 - epsilon) * policy + epsilon * noise`` (on legal actions)

    Ref:
        https://datascienceschool.net/02%20mathematics/08.07%20%EB%B2%A0%ED%83%80%EB%B6%84%ED%8F%AC%2C%20%EA%B0%90%EB%A7%88%EB%B6%84%ED%8F%AC%2C%20%EB%94%94%EB%A6%AC%ED%81%B4%EB%A0%88%20%EB%B6%84%ED%8F%AC.html#id6
    """
    if epsilon <= 0.0 or not torch.any(legal_mask):
        return policy

    # Align legal_mask dimensions to policy for broadcasting
    if policy.dim() > legal_mask.dim():
        while legal_mask.dim() < policy.dim():
            legal_mask = legal_mask.unsqueeze(0)

    num_legal = int(legal_mask.sum().item())
    if num_legal <= 1:
        return policy

    noise = torch.from_numpy(np.random.dirichlet([alpha] * num_legal)).to(
        device=policy.device, dtype=policy.dtype
    )
    expanded = torch.zeros_like(policy)
    expanded[legal_mask] = noise

    mixed = (1.0 - epsilon) * policy + epsilon * expanded
    mixed *= legal_mask
    total = mixed.sum()

    if total > 1e-6:
        mixed /= total
    else:
        mixed[legal_mask] = 1.0 / num_legal

    return mixed


class SearchEngine(ABC):
    """Abstract search engine that encapsulates selection/expansion/backprop flow."""

    def __init__(
        self,
        game: Gomoku,
        mcts_params: MctsConfig,
        inference_client: InferenceClient,
        batch_size: int | None = None,
        min_batch_size: int | None = None,
        max_wait_ms: int | None = None,
        async_inflight_limit: int | None = None,
        use_fp16: bool = True,
    ):
        self.game = game
        self.params = mcts_params
        self.inference = inference_client
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.max_wait_ms = max_wait_ms
        self.async_inflight_limit = async_inflight_limit
        self.use_fp16 = bool(use_fp16)

    @abstractmethod
    def search(self, roots: list[TreeNode], add_noise: bool = False) -> None:
        """Run MCTS simulations starting from the given roots."""

    # --- Helpers reused across engines ---

    def _seed_random_generators(self) -> None:
        """Reseed RNGs per process to avoid identical rollouts."""
        seed = int.from_bytes(os.urandom(4), byteorder="little")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _legal_indices(self, state: GameState) -> np.ndarray:
        """
        Return legal move indices, leveraging the state's internal cache.

        Notes
        -----
        - Populates `state.legal_indices_cache` via `game.get_legal_moves` if None.
        - Returns int64 array for consistency with PyTorch/Indices requirements.
        - Distinction: None means 'uncalculated', Empty array means 'no moves'.
        """
        # 1. Update Cache if Uninitialized (Lazy Loading)
        if state.legal_indices_cache is None:
            # Side-effect: This populates state.legal_indices_cache
            self.game.get_legal_moves(state)

        cache = state.legal_indices_cache

        # 2. Safety Check: If still None (buggy logic) or Empty (terminal state)
        if cache is None or cache.size == 0:
            return np.array([], dtype=np.int64)

        # 3. Cast to int64 (Copy occurs here due to int16 -> int64)
        return cache.astype(np.int64, copy=False)

    def _legal_mask_numpy(self, state: GameState) -> np.ndarray:
        """Return a boolean mask of legal moves for the given state."""
        mask = np.zeros(self.game.action_size, dtype=bool)
        indices = self._legal_indices(state)
        if indices.size:
            mask[indices] = True
        return mask

    def _legal_mask_tensor(
        self, state: GameState, device: torch.device
    ) -> torch.Tensor:
        """Return a boolean tensor mask of legal moves for the given state."""
        mask_np = self._legal_mask_numpy(state)
        if not mask_np.size:
            return torch.zeros(self.game.action_size, dtype=torch.bool, device=device)
        idx = torch.from_numpy(mask_np.nonzero()[0]).to(device=device, dtype=torch.long)
        mask = torch.zeros(self.game.action_size, dtype=torch.bool, device=device)
        mask.index_fill_(0, idx, True)
        return mask

    def _encode_state_to_tensor(
        self, state: GameState, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """
        Encode a single game state into a tensor for inference.

        Notes
        -----
        - Prefer native core encoding when ``game.use_native`` and ``state.native_state``
          are available to skip Python-side feature construction.
        - Falls back to Python encoder on any failure to keep inference robust.
        """
        if (
            getattr(self.game, "use_native", False)
            and getattr(self.game, "_native_core", None) is not None
            and getattr(state, "native_state", None) is not None
        ):
            try:
                encoded_np = self.game._native_core.write_state_features(  # noqa: SLF001
                    state.native_state
                )
                return torch.as_tensor(encoded_np, dtype=dtype, device=device)
            except Exception:  # noqa: BLE001
                # Fallback to Python path if native encoding fails for any reason.
                pass

        encoded_np = self.game.get_encoded_state([state])[0]
        return torch.as_tensor(encoded_np, dtype=dtype, device=device)

    def _masked_policy_from_logits(
        self,
        logits: torch.Tensor,
        legal_mask: torch.Tensor,
        apply_dirichlet: bool = False,
    ) -> torch.Tensor:
        """Apply legal masking and optional Dirichlet noise to logits."""
        if not torch.any(legal_mask):
            return torch.zeros_like(logits, dtype=logits.dtype)
        masked = torch.where(
            legal_mask,
            logits,
            torch.tensor(-float("inf"), device=logits.device, dtype=logits.dtype),
        )
        policy = F.softmax(masked, dim=-1)
        if apply_dirichlet:
            policy = _safe_dirichlet_noise(
                policy,
                legal_mask,
                float(self.params.dirichlet_epsilon),
                float(self.params.dirichlet_alpha),
            )
        return policy

    def _expand_with_policy(self, node: TreeNode, policy: torch.Tensor) -> None:
        # Accept either [A] or [B, A]; TreeNode.expand expects 1D.
        if policy.dim() == 2 and policy.size(0) == 1:
            policy = policy.squeeze(0)

        expected_size = self.game.action_size
        if policy.numel() != expected_size:
            raise ValueError(
                f"Policy dimension error! Expected {expected_size} elements (1 state), "
                f"but got {policy.numel()} elements with shape {policy.shape}. "
                "Did you accidentally pass a batch to a single-node expansion?"
            )
        legal_mask_np = self._legal_mask_numpy(node.state)
        policy_np = policy.detach().cpu().numpy()
        node.expand(policy_np, legal_mask_np.astype(np.float32), self.game)

    def _backup(self, node: TreeNode, value: float) -> None:
        node.backup(value)

    def _evaluate_terminal(self, node: TreeNode) -> tuple[float, bool]:
        """
        Evaluate terminal status and return value from node owner's perspective.

        Delegates to TreeNode.get_terminal_info() for caching.

        Parameters
        ----------
        node : TreeNode
            Node to evaluate.

        Returns
        -------
        tuple[float, bool]
            (value, is_terminal) where value is from the node owner's perspective.
        """
        return node.get_terminal_info(self.game)

    def _ensure_root_noise(self, root: TreeNode) -> None:
        """
        Inject Dirichlet noise into an already expanded root (tree reuse guard).

        Notes
        -----
        - 중복 주입을 막기 위해 root.noise_applied 플래그를 확인한다.
        - 루트가 아직 확장되지 않았거나 epsilon이 0이면 아무 것도 하지 않는다.
        - 트리 재사용 시 탐험 손실을 줄이기 위해 children 유무와 관계없이
          루트 정책 전체에 노이즈를 적용 후 재확장한다.
        """
        if getattr(root, "noise_applied", False):
            return
        if float(self.params.dirichlet_epsilon) <= 0.0:
            return
        legal_mask_np = self._legal_mask_numpy(root.state)
        if legal_mask_np.sum() == 0:
            return

        # 현재 루트 정책을 children이 있으면 visit 기반으로, 없으면 균등 분포로 구성
        if root.children:
            priors_np = np.zeros(self.game.action_size, dtype=np.float32)
            for child in root.children.values():
                if child.action_taken:
                    idx = (
                        child.action_taken[0]
                        + child.action_taken[1] * self.game.col_count
                    )
                    priors_np[idx] = max(child.prior, 0.0)
            if priors_np.sum() <= 0:
                priors_np[legal_mask_np] = 1.0
        else:
            priors_np = np.zeros(self.game.action_size, dtype=np.float32)
            priors_np[legal_mask_np] = 1.0

        priors_t = torch.tensor(priors_np, dtype=torch.float32, device="cpu")
        legal_mask_t = torch.tensor(legal_mask_np, dtype=torch.bool, device="cpu")

        noisy_policy = (
            _safe_dirichlet_noise(
                priors_t,
                legal_mask_t,
                epsilon=float(self.params.dirichlet_epsilon),
                alpha=float(self.params.dirichlet_alpha),
            )
            .cpu()
            .numpy()
        )

        # children이 없을 때는 노이즈가 적용된 정책으로 재확장
        if not root.children:
            legal_mask = legal_mask_np.astype(np.float32)
            root.expand(noisy_policy, legal_mask, self.game)
        else:
            for child in root.children.values():
                if child.action_taken:
                    idx = (
                        child.action_taken[0]
                        + child.action_taken[1] * self.game.col_count
                    )
                    child.prior = float(noisy_policy[idx])

        root.noise_applied = True

    def get_search_result(self, root: TreeNode) -> tuple[np.ndarray, dict]:
        """
        Compute policy and stats from a searched root node.

        Parameters
        ----------
        root : TreeNode
            Root node after search has completed.

        Returns
        -------
        tuple[np.ndarray, dict]
            Visit-based policy and statistics dictionary.
        """
        visits_policy = np.zeros(self.game.action_size, dtype=np.float32)
        policy = np.zeros(self.game.action_size, dtype=np.float32)
        total_visits_policy = 0

        if root.children:
            total_visits_policy = max(0, root.visit_count - 1)
            if total_visits_policy > 0:
                for child in root.children.values():
                    if child.action_taken:
                        idx = (
                            child.action_taken[0]
                            + child.action_taken[1] * self.game.col_count
                        )
                        visits_policy[idx] = child.visit_count
                if visits_policy.sum() > 0:
                    policy = visits_policy / visits_policy.sum()

        if total_visits_policy == 0 or np.sum(policy) < 1e-6:
            legal_mask_policy = self._legal_mask_numpy(root.state).astype(np.float32)
            num_legal_policy = legal_mask_policy.sum()
            policy = (
                legal_mask_policy / num_legal_policy
                if num_legal_policy > 0
                else np.ones(self.game.action_size, dtype=np.float32)
                / self.game.action_size
            )

        q_max = -float("inf")
        q_selected = 0.0
        child_stats: list[dict] = []
        visits_stats = np.zeros(self.game.action_size)
        q_values_stats = np.full(self.game.action_size, -float("inf"))

        if root.children:
            for child in root.children.values():
                if child.action_taken:
                    idx = (
                        child.action_taken[0]
                        + child.action_taken[1] * self.game.col_count
                    )
                    visits_stats[idx] = child.visit_count
                    if child.visit_count > 0:
                        q_root = -(child.value_sum / child.visit_count)
                        wp = (q_root + 1.0) / 2.0
                    else:
                        q_root, wp = 0.0, 0.5

                    child_stats.append(
                        {
                            "idx": idx,
                            "visit": child.visit_count,
                            "q_root": q_root,
                            "wp": wp,
                        }
                    )
                    q_values_stats[idx] = q_root
                    q_max = max(q_max, q_root)

            if np.sum(visits_stats) > 0:
                best_visit_idx_np = np.argmax(visits_stats)
                if q_values_stats[best_visit_idx_np] > -float("inf"):
                    q_selected = q_values_stats[best_visit_idx_np]
                elif child_stats:
                    q_selected = child_stats[0]["q_root"]
            elif child_stats:
                q_selected = child_stats[0]["q_root"]

        stats = {
            "q_max": q_max if q_max > -float("inf") else 0.0,
            "q_selected": q_selected,
            "child_stats": sorted(child_stats, key=lambda x: x["visit"], reverse=True),
        }
        return policy, stats

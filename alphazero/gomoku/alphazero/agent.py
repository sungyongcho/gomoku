"""Agent module wrapping PVMCTS with a clean self-play interface."""

from collections.abc import Sequence

import numpy as np

from gomoku.alphazero.types import (
    Action,
    PolicyVector,
    action_to_xy,
)
from gomoku.core.gomoku import GameState, Gomoku
from gomoku.inference.base import InferenceClient
from gomoku.pvmcts.pvmcts import PVMCTS
from gomoku.pvmcts.treenode import TreeNode
from gomoku.utils.config.loader import MctsConfig


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature to a policy distribution.

    Parameters
    ----------
    probs : np.ndarray
        Policy probability vector.
    temperature : float
        Sampling temperature. Values <= 0 collapse to deterministic argmax.

    Returns
    -------
    np.ndarray
        Temperature-adjusted probability vector.
    """
    # Guard against near-zero or negative temperatures
    temp = max(float(temperature), 1e-8)
    # Deterministic argmax when temperature is effectively zero
    if temp <= 1e-8:
        one_hot = np.zeros_like(probs, dtype=np.float32)
        one_hot[int(np.argmax(probs))] = 1.0
        return one_hot
    # Keep distribution when temperature is effectively 1
    if abs(temp - 1.0) < 1e-6:
        return probs
    exponent = 1.0 / temp
    safe = np.maximum(probs, 1e-12) ** exponent
    safe_sum = safe.sum()
    # Fallback to argmax on pathological sums
    if safe_sum <= 0 or np.isnan(safe_sum) or np.isinf(safe_sum):
        one_hot = np.zeros_like(probs, dtype=np.float32)
        one_hot[int(np.argmax(probs))] = 1.0
        return one_hot
    safe /= safe_sum
    return safe.astype(np.float32)


class AlphaZeroAgent:
    """Unified agent that drives PVMCTS for single or batched self-play."""

    def __init__(
        self,
        game: Gomoku,
        mcts_cfg: MctsConfig,
        inference_client: InferenceClient,
        engine_type: str = "sequential",
        async_inflight_limit: int | None = None,
    ):
        """Initialize the agent and underlying PVMCTS instance.

        Parameters
        ----------
        game : Gomoku
            Game engine instance.
        mcts_cfg : MctsConfig
            MCTS configuration.
        inference_client : InferenceClient
            Backend used for policy/value inference.
        engine_type : str, optional
            Search engine type (sequential/vectorize/mp/ray).
            Defaults to ``"sequential"``.
        async_inflight_limit : int | None, optional
            Optional limit for async inflight requests (ray engine).
        """
        self.game: Gomoku = game
        self.mcts_cfg: MctsConfig = mcts_cfg
        self.inference: InferenceClient = inference_client
        self.engine_type: str = engine_type
        self.async_inflight_limit: int | None = async_inflight_limit
        self.roots: list[TreeNode | None] = []
        self._last_value: float = 0.0  # For adjudication
        self.mcts = PVMCTS(
            game,
            mcts_cfg,
            inference_client,
            mode=engine_type,
            async_inflight_limit=async_inflight_limit,
        )

    @property
    def root(self) -> TreeNode | None:
        """Single-slot root view kept for backwards compatibility."""
        return self.roots[0] if self.roots else None

    def reset(self) -> None:
        """Clear cached roots."""
        self.roots = []

    def reset_game(self, slot_idx: int) -> None:
        """Reset the cached root for a specific slot."""
        if 0 <= slot_idx < len(self.roots):
            self.roots[slot_idx] = None

    def _state_matches_root(self, root_state: GameState, state_raw: GameState) -> bool:
        """Check whether a cached root state matches the given state."""
        return (
            root_state.next_player == state_raw.next_player
            and root_state.last_move_idx == state_raw.last_move_idx
            and root_state.p1_pts == state_raw.p1_pts
            and root_state.p2_pts == state_raw.p2_pts
            and root_state.empty_count == state_raw.empty_count
            and root_state.history == state_raw.history
        )

    def _ensure_roots(self, states_raw: Sequence[GameState]) -> list[TreeNode]:
        """Ensure cached roots align with the provided states."""
        if len(self.roots) != len(states_raw):
            self.roots = [None] * len(states_raw)

        roots: list[TreeNode] = []
        for idx, state in enumerate(states_raw):
            root = self.roots[idx]
            if root is None or not self._state_matches_root(root.state, state):
                root = self.mcts.create_root(state)
                self.roots[idx] = root
            roots.append(root)
        return roots

    def _normalize_temperatures(
        self, temperature: float | Sequence[float], count: int
    ) -> list[float]:
        """Normalize temperature argument to a list matching the number of states."""
        if isinstance(temperature, Sequence) and not isinstance(
            temperature, (str, bytes)
        ):
            temps = [float(t) for t in temperature]
            if len(temps) != count:
                raise ValueError(
                    "temperature length must match number of states: "
                    f"{len(temps)} != {count}"
                )
            return [max(t, 1e-8) for t in temps]
        return [max(float(temperature), 1e-8)] * count

    def _normalize_noise_flags(
        self, add_noise_flags: bool | Sequence[bool], count: int
    ) -> list[bool]:
        """Normalize noise flags to a list matching the number of states."""
        if isinstance(add_noise_flags, Sequence) and not isinstance(
            add_noise_flags, (str, bytes)
        ):
            flags = [bool(v) for v in add_noise_flags]
            if len(flags) != count:
                raise ValueError(
                    "add_noise_flags length must match number of states: "
                    f"{len(flags)} != {count}"
                )
            return flags
        return [bool(add_noise_flags)] * count

    def _run_search_group(
        self, roots: list[TreeNode], add_noise: bool
    ) -> list[np.ndarray]:
        """Run search for a group of roots and return policies."""
        results = self.mcts.run_search(roots, add_noise=add_noise)
        # Capture value for adjudication (from first/single root)
        if results and len(results) > 0:
            _, stats = results[0]
            # Try to extract value from stats
            if isinstance(stats, dict):
                self._last_value = float(stats.get("value", stats.get("q_max", 0.0)))
            else:
                self._last_value = 0.0
        return [policy for policy, _ in results]

    def get_action_probs(
        self,
        state_raw: GameState,
        temperature: float,
        add_noise: bool,
    ) -> PolicyVector:
        """Return a policy distribution for a single state."""
        policies = self.get_action_probs_batch(
            [state_raw], temperature=temperature, add_noise_flags=[add_noise]
        )
        return policies[0]

    def get_action_probs_batch(
        self,
        states_raw: Sequence[GameState],
        temperature: float | Sequence[float],
        add_noise_flags: bool | Sequence[bool],
        active_indices: list[int] | None = None,
    ) -> list[PolicyVector]:
        """Return policy distributions for a batch of states.

        Parameters
        ----------
        states_raw : Sequence[GameState]
            States to evaluate.
        temperature : float | Sequence[float]
            Temperature per state or a scalar applied to all.
        add_noise_flags : bool | Sequence[bool]
            Flags indicating whether to add Dirichlet noise per state.
        active_indices : list[int] | None, optional
            If provided, only states at these indices will be searched.
            Dummies are returned for others. This helps keep `roots` synchronized
            with a larger external batch.

        Returns
        -------
        list[PolicyVector]
            Temperature-adjusted policy distributions.
        """
        if not states_raw:
            return []

        # 1. Ensure internal roots matches the full states_raw size
        roots = self._ensure_roots(states_raw)

        # 2. Extract valid indices and temps
        active = (
            active_indices
            if active_indices is not None
            else list(range(len(states_raw)))
        )
        if not active:
            # Still return a list of zeroes of correct length if none active
            dummy = np.zeros(self.game.action_size, dtype=np.float32)
            return [dummy] * len(states_raw)

        temps = self._normalize_temperatures(temperature, len(states_raw))
        noise_flags = self._normalize_noise_flags(add_noise_flags, len(states_raw))

        results: list[PolicyVector | None] = [None] * len(states_raw)

        # 3. Only search for active indices
        for flag in (True, False):
            # Intersection of 'active' and those matching 'flag'
            idxs = [i for i in active if noise_flags[i] is flag]
            if not idxs:
                continue
            group_roots = [roots[i] for i in idxs]
            policies = self._run_search_group(group_roots, add_noise=flag)
            for idx, policy in zip(idxs, policies, strict=True):
                results[idx] = _apply_temperature(policy, temps[idx])

        # 4. Fill in dummies for inactive indices
        dummy_policy = np.zeros(self.game.action_size, dtype=np.float32)
        final_results = []
        for r in results:
            if r is not None:
                final_results.append(r)
            else:
                final_results.append(dummy_policy)

        return final_results

    def update_root(self, action: Action) -> None:
        """Update the single-slot root after an action."""
        if len(self.roots) != 1:
            if self.roots:
                self.roots = [self.roots[0]]
            else:
                self.roots = [None]
        self.update_root_batch([action])

    def update_root_batch(self, actions: Sequence[Action]) -> None:
        """Update cached roots for a batch of actions."""
        if not self.roots:
            return

        if len(actions) != len(self.roots):
            raise ValueError(
                "actions length must match number of roots: "
                f"{len(actions)} != {len(self.roots)}"
            )

        for idx, action in enumerate(actions):
            root = self.roots[idx]
            if root is None:
                continue
            x, y = action_to_xy(action, self.game.col_count)
            child = root.children.get((x, y))
            if child is not None:
                self.roots[idx] = child
                child.parent = None
            else:
                self.roots[idx] = None

    def get_last_value(self) -> float:
        """Return the value estimate from the last search (for adjudication)."""
        return self._last_value

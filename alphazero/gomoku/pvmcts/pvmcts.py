import numpy as np
import torch

from gomoku.core.gomoku import GameState, Gomoku
from gomoku.inference.base import InferenceClient
from gomoku.pvmcts.search.engine import SearchEngine
from gomoku.pvmcts.search.mp import MultiprocessEngine
from gomoku.pvmcts.search.sequential import SequentialEngine
from gomoku.pvmcts.search.vectorize import VectorizeEngine
from gomoku.pvmcts.treenode import TreeNode
from gomoku.utils.config.loader import MctsConfig


def _safe_cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except (AssertionError, RuntimeError, AttributeError):
        return False


class PVMCTS:
    """
    Facade for PUCT-based MCTS that delegates search to a pluggable SearchEngine.

    Modes supported: sequential, batch, mp(placeholder), ray_async.
    """

    def __init__(
        self,
        game: Gomoku,
        mcts_params: MctsConfig,
        inference_client: InferenceClient,
        mode: str | None = None,
        async_inflight_limit: int | None = None,
    ):
        self.game = game
        self.params = mcts_params
        self.inference = inference_client
        self.device = self._resolve_device(inference_client)
        self.async_inflight_limit = self._resolve_async_limit(
            inference_client, async_inflight_limit
        )
        self.mode = (mode or self._infer_mode()).lower()
        self.engine: SearchEngine = self._build_engine()

    def _resolve_device(self, client: InferenceClient) -> torch.device:
        cuda_available = _safe_cuda_available()
        device = getattr(client, "device", None)
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            try:
                return torch.device(device)
            except Exception:
                return torch.device("cuda" if cuda_available else "cpu")
        try:
            return torch.device("cuda" if cuda_available else "cpu")
        except Exception:
            return torch.device("cpu")

    def _resolve_async_limit(
        self, client: InferenceClient, override: int | None
    ) -> int | None:
        inferred = (
            override
            if override is not None
            else getattr(client, "async_inflight_limit", None)
        )
        if inferred is not None and inferred <= 0:
            return None
        return inferred

    def _infer_mode(self) -> str:
        if hasattr(self.inference, "infer_async") and callable(
            self.inference.infer_async
        ):
            return "ray"
        if getattr(self.device, "type", "cpu") == "cpu":
            return "sequential"
        return "vectorize"

    def _build_engine(self) -> SearchEngine:
        batch_size = self.params.batch_infer_size
        min_batch = self.params.min_batch_size
        max_wait = self.params.max_batch_wait_ms
        use_fp16 = getattr(self.inference, "device", None)
        use_fp16 = bool(isinstance(use_fp16, torch.device) and use_fp16.type == "cuda")

        mode = self.mode
        use_native = getattr(self.params, "use_native", False)
        if mode not in {"sequential", "vectorize", "mp", "ray"}:
            raise ValueError(
                f"Unsupported mode '{mode}'. Use one of: sequential, vectorize, mp, ray."
            )
        if mode == "ray":
            if use_native:
                # SequentialEngine + use_native supports async infer (Ray) via CppSearchStrategy
                return SequentialEngine(
                    self.game,
                    self.params,
                    self.inference,
                    use_native=True,
                )
            try:
                from gomoku.pvmcts.search.ray.ray_async import RayAsyncEngine
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Ray mode requested, but Ray dependencies are not installed. "
                    "Install with extras: pip install -e \".[ray]\""
                ) from exc
            return RayAsyncEngine(
                self.game,
                self.params,
                self.inference,
                batch_size=batch_size,
                min_batch_size=min_batch,
                max_wait_ms=max_wait,
                async_inflight_limit=self.async_inflight_limit,
                use_fp16=use_fp16,
            )

        if use_native:
            if mode == "vectorize":
                raise ValueError("Native MCTS does not support 'vectorize' mode yet.")
            return SequentialEngine(
                self.game,
                self.params,
                self.inference,
                use_native=True,
            )
        if mode == "sequential":
            return SequentialEngine(self.game, self.params, self.inference)
        if mode == "mp":
            return MultiprocessEngine(
                self.game,
                self.params,
                self.inference,
                batch_size=batch_size,
                min_batch_size=min_batch,
                max_wait_ms=max_wait,
                use_fp16=use_fp16,
            )
        if mode == "vectorize":
            return VectorizeEngine(
                self.game,
                self.params,
                self.inference,
                batch_size=batch_size,
                min_batch_size=min_batch,
                max_wait_ms=max_wait,
                use_fp16=use_fp16,
            )
        raise ValueError(
            f"Unsupported mode '{mode}'. Use one of: sequential, vectorize, mp, ray."
        )

    def create_root(self, state: GameState) -> TreeNode:
        """Create a root node for the given state."""
        return TreeNode(state=state)

    def run_search(
        self, roots: list[TreeNode], add_noise: bool = True
    ) -> list[tuple[np.ndarray, dict]]:
        """
        Unified search API for single or multiple roots.

        Parameters
        ----------
        roots : list[TreeNode]
            Root nodes to search. Empty list returns [].
        add_noise : bool
            Whether to apply Dirichlet noise on initial expansion/root reuse.

        Returns
        -------
        list[tuple[np.ndarray, dict]]
            (policy, stats) per root.
        """
        if not roots:
            return []
        self.engine.search(roots, add_noise=add_noise)
        return [self.engine.get_search_result(root) for root in roots]

    @torch.no_grad()
    def search(self, state: GameState) -> np.ndarray:
        """Run MCTS on the given state and return the policy distribution."""
        root = self.create_root(state)
        policy, _ = self.run_search([root], add_noise=True)[0]
        return policy

    @torch.no_grad()
    def search_with_root_stats(self, state: GameState) -> tuple[np.ndarray, dict]:
        """Run MCTS and return the policy distribution with root stats."""
        root = self.create_root(state)
        return self.run_search([root], add_noise=True)[0]

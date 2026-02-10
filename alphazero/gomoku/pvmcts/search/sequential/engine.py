import numpy as np
import torch
from gomoku.core.gomoku import GameState, Gomoku
from gomoku.inference.base import InferenceClient
from gomoku.pvmcts.search.engine import SearchEngine
from gomoku.pvmcts.treenode import TreeNode
from gomoku.utils.config.loader import MctsConfig

from gomoku.pvmcts.search.sequential.python_strategy import PythonSearchStrategy
from gomoku.pvmcts.search.sequential.cpp_strategy import CppSearchStrategy

class SequentialEngine(SearchEngine):
    """Facade for sequential search that delegates to Python or Native strategies."""

    def __init__(
        self,
        game: Gomoku,
        mcts_params: MctsConfig,
        inference_client: InferenceClient,
        use_native: bool = False,
        **kwargs,
    ):
        super().__init__(game, mcts_params, inference_client, **kwargs)
        self.use_native = bool(use_native)
        self._native_engine = None
        self._inference_device = self._resolve_inference_device(inference_client)

        if self.use_native:
            if getattr(game, "_native_core", None) is None:
                raise ValueError("Gomoku(use_native=True) requires a native core.")

            try:
                from gomoku.cpp_ext import gomoku_cpp  # type: ignore
            except Exception as exc:
                raise RuntimeError("gomoku_cpp extension is not available; rebuild the package.") from exc

            self._native_engine = gomoku_cpp.MctsEngine(game._native_core, float(mcts_params.C))
            self.strategy = CppSearchStrategy()
        else:
            self.strategy = PythonSearchStrategy()

    def search(self, roots: TreeNode | list[TreeNode], add_noise: bool = False) -> None:
        """Run sequential MCTS simulations starting from the given roots."""
        if isinstance(roots, TreeNode):
            roots = [roots]
        if not roots:
            return

        self.strategy.search(self, roots, add_noise)

    def _resolve_inference_device(self, inference_client: InferenceClient) -> torch.device:
        device_attr = getattr(inference_client, "device", None)
        if isinstance(device_attr, torch.device): return device_attr
        if isinstance(device_attr, str):
            try: return torch.device(device_attr)
            except Exception: return torch.device("cpu")
        try: return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception: return torch.device("cpu")

    def _infer_single(self, node_state, dtype: torch.dtype, device: torch.device, apply_dirichlet: bool) -> tuple[torch.Tensor, float]:
        state_tensor = self._encode_state_to_tensor(node_state, dtype=dtype, device=device)
        logits, value_tensor = self.inference.infer(state_tensor)
        if logits.dim() == 1: logits = logits.unsqueeze(0)
        value = float(value_tensor.squeeze().item())
        legal_mask = self._legal_mask_tensor(node_state, logits.device)
        policy = self._masked_policy_from_logits(logits, legal_mask, apply_dirichlet=apply_dirichlet)
        return policy, value

    def _evaluate_terminal(self, node: TreeNode) -> tuple[float, bool]:
        # Helper to avoid repetition in strategies if desired
        # (Though SearchEngine already has some of these)
        # For simplicity, strategies can call these methods on engine.
        return super()._evaluate_terminal(node)

    # Re-expose or wrap other methods from SearchEngine as needed by strategies

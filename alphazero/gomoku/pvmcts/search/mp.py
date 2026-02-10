import os
import sys

import torch

from gomoku.core.gomoku import Gomoku
from gomoku.inference.mp_client import MPInferenceClient
from gomoku.pvmcts.search.vectorize import VectorizeEngine
from gomoku.pvmcts.treenode import TreeNode
from gomoku.utils.config.loader import MctsConfig


class MultiprocessEngine(VectorizeEngine):
    """
    Vectorize-based search engine that delegates inference over multiprocessing IPC.

    Notes
    -----
    - Batches leaf nodes and sends a single CPU tensor to ``MPInferenceClient``.
    - Reseeds random generators per process to keep search diversity.
    - Wraps search with basic BrokenPipe/EOF handling to avoid crashing workers.
    """

    def __init__(
        self,
        game: Gomoku,
        params: MctsConfig,
        inference: MPInferenceClient,
        batch_size: int | None = None,
        min_batch_size: int | None = None,
        max_wait_ms: int | None = None,
        use_fp16: bool = False,
    ):
        super().__init__(
            game,
            params,
            inference,
            batch_size=batch_size,
            min_batch_size=min_batch_size,
            max_wait_ms=max_wait_ms,
            async_inflight_limit=None,
            use_fp16=use_fp16,
        )
        self.pid = os.getpid()

        self._seed_random_generators()

    def search(
        self, start_nodes: TreeNode | list[TreeNode], add_noise: bool = False
    ) -> None:
        """
        Run batched search with IPC-safe error handling.

        Parameters
        ----------
        start_nodes : TreeNode or list[TreeNode]
            Starting node(s) to search.
        add_noise : bool
            Whether to apply Dirichlet noise on initial roots.
        """
        try:
            super().search(start_nodes, add_noise=add_noise)
        except (BrokenPipeError, EOFError, ConnectionResetError):
            print(
                f"[MultiprocessEngine {self.pid}] Inference Server connection lost.",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"[MultiprocessEngine {self.pid}] Unexpected error: {e}",
                file=sys.stderr,
            )
            raise e

    def _process_batch(
        self, nodes: list[TreeNode], is_start_node_batch: bool, add_noise: bool
    ) -> None:
        """
        Encode nodes, run MP inference on CPU, and scatter results back.

        Parameters
        ----------
        nodes : list[TreeNode]
            Leaf nodes to evaluate and expand.
        is_start_node_batch : bool
            Whether this batch corresponds to initial root expansion (Dirichlet noise).
        """
        if not nodes:
            return

        device = torch.device("cpu")
        dtype = torch.float32

        batch_tensors = [
            self._encode_state_to_tensor(n.state, dtype=dtype, device=device)
            for n in nodes
        ]
        input_tensor = torch.stack(batch_tensors)

        policy_logits, values = self.inference.infer(input_tensor)

        if policy_logits.size(0) != len(nodes) or values.size(0) != len(nodes):
            raise RuntimeError(
                f"Batch size mismatch! Requested {len(nodes)}, "
                f"got logits {policy_logits.size(0)}, values {values.size(0)}"
            )

        policy_logits = policy_logits.cpu()
        values = values.view(-1).cpu()

        for i, node in enumerate(nodes):
            val_scalar = float(values[i].item())
            logits = policy_logits[i]

            legal_mask = self._legal_mask_tensor(node.state, logits.device)
            policy = self._masked_policy_from_logits(
                logits,
                legal_mask,
                apply_dirichlet=is_start_node_batch and add_noise,
            )

            self._expand_with_policy(node, policy)
            self._backup(node, val_scalar)

    def __repr__(self) -> str:
        return f"<MultiprocessEngine(pid={self.pid})>"

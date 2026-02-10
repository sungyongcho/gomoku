# =============================================================================
# Educational / reference implementation.
# Demonstrates the bare-minimum multiprocessing pattern (queue send & blocking wait)
# without batching. This is intentionally inefficient; do not use in production.
# -----------------------------------------------------------------------------
# What this file shows (intentionally verbose):
#   - How to start worker processes with `spawn` so PyTorch models load safely.
#   - How to pass encoded states over a SimpleQueue and block until the result
#     returns (no overlap, no batching).
#   - How to keep tree mutation (selection/backup) only in the main process and
#     delegate inference to workers.
#   - Why this is slow: every inference waits on IPC; only one leaf is in flight.
# -----------------------------------------------------------------------------
# =============================================================================


import multiprocessing as mp
import os

import numpy as np
import torch

from gomoku.pvmcts.search.engine import SearchEngine
from gomoku.pvmcts.treenode import TreeNode


def _worker_sequential_loop(rank, task_q, result_q, inference_factory, device_str):
    """Worker process entry: consume one request at a time and return results."""
    # Each worker owns its own inference client; the factory must be picklable.
    inference_client = inference_factory()
    device = torch.device(device_str)

    if hasattr(inference_client, "to"):
        inference_client.to(device)

    while True:
        # Wait for work (blocking). Sentinel None signals shutdown.
        item = task_q.get()
        if item is None:
            break

        request_id, state_np = item

        with torch.no_grad():
            # Convert NumPy state to tensor on the worker's device.
            tensor_input = torch.as_tensor(state_np, dtype=torch.float32, device=device)

            if hasattr(inference_client, "infer_batch"):
                if tensor_input.dim() == 3:
                    tensor_input = tensor_input.unsqueeze(0)
                logits, value = inference_client.infer_batch(tensor_input)
            else:
                logits, value = inference_client.infer(tensor_input)

        # Move to CPU before sending over the queue.
        result_q.put(
            (
                request_id,
                logits.detach().cpu().numpy(),
                value.detach().cpu().numpy(),
            )
        )


class MultiprocessSequentialEngine(SearchEngine):
    """Queue-based engine that offloads inference to worker processes."""

    def __init__(
        self,
        game,
        params,
        inference_factory,
        num_workers=None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(game, params, None, **kwargs)

        # Keep worker count modest; oversubscription only adds IPC overhead.
        self.num_workers = num_workers or max(1, os.cpu_count() // 2)

        # Use spawn (safer with PyTorch/CUDA than fork).
        ctx = mp.get_context("spawn")
        self.task_q = ctx.SimpleQueue()
        self.result_q = ctx.SimpleQueue()

        self.workers = []
        for i in range(self.num_workers):
            p = ctx.Process(
                target=_worker_sequential_loop,
                args=(i, self.task_q, self.result_q, inference_factory, str(device)),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

        self._seed_random_generators()

    def _seed_random_generators(self):
        """Set per-process seeds to keep randomness diverged across workers."""
        seed = int.from_bytes(os.urandom(4), byteorder="little")
        np.random.seed(seed)
        torch.manual_seed(seed)

    def search(self, roots: TreeNode | list[TreeNode], add_noise: bool = False) -> None:
        """MCTS loop that blocks on every inference call (no batching)."""
        if isinstance(roots, TreeNode):
            roots = [roots]
        for root in roots:
            if root is None:
                continue
            self._search_single(root, add_noise=add_noise)

    def _search_single(self, root: TreeNode, add_noise: bool) -> None:
        target_searches = int(self.params.num_searches)
        if target_searches <= 0:
            return

        if root.visit_count == 0:
            self._evaluate_and_expand(root, is_start_node=True and add_noise)
        elif add_noise and root.children:
            self._ensure_root_noise(root)

        for _ in range(target_searches):
            node = root

            # Selection: descend using UCB until a leaf is reached.
            while not node.is_leaf:
                node = node.best_child(self.params.C)

            value, is_terminal = self._evaluate_terminal(node)

            if not is_terminal:
                # Offload one leaf to a worker and wait synchronously.
                value = self._evaluate_and_expand(node, is_start_node=False)

            node.backup(value)

    def _evaluate_and_expand(self, node: TreeNode, is_start_node: bool) -> float:
        """Submit a single node for inference and expand after the response."""
        encoded = self.game.get_encoded_state([node.state])[0]

        req_id = 0
        # Only one outstanding request is supported; this is purely sequential IPC.
        self.task_q.put((req_id, encoded))

        _, logits_np, value_np = self.result_q.get()

        val_scalar = float(value_np.item())
        logits = torch.as_tensor(logits_np).squeeze()

        legal_mask = self._legal_mask_tensor(node.state, logits.device)
        policy = self._masked_policy_from_logits(
            logits,
            legal_mask,
            apply_dirichlet=is_start_node,
        )

        self._expand_with_policy(node, policy)

        return val_scalar

    def close(self):
        for _ in self.workers:
            self.task_q.put(None)
        for p in self.workers:
            p.join(timeout=1)

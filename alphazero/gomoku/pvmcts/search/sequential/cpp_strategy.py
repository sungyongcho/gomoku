
import numpy as np
import torch
import threading
import queue
import time
from typing import TYPE_CHECKING, List, Tuple
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from gomoku.pvmcts.search.sequential.engine import SequentialEngine
    from gomoku.pvmcts.treenode import TreeNode

class SearchStrategy(ABC):
    """Abstract base class for search strategies."""
    @abstractmethod
    def search(self, engine: 'SequentialEngine', roots: List['TreeNode'], add_noise: bool) -> None:
        pass

class CppSearchStrategy(SearchStrategy):
    """C++ Native implementation of sequential MCTS with threaded async pipelining."""

    def search(self, engine: 'SequentialEngine', roots: List['TreeNode'], add_noise: bool) -> None:
        if engine._native_engine is None:
            raise RuntimeError("Native engine is not initialized.")

        for root in roots:
            if root is None:
                continue
            self._run_native_mcts(engine, root, add_noise)

    def _run_native_mcts(self, engine: 'SequentialEngine', root: 'TreeNode', add_noise: bool) -> None:
        """Execute C++ MCTS and sync results to the Python tree."""
        sims = int(engine.params.num_searches)
        if sims <= 0:
            return

        if root.state.native_state is None:
            raise RuntimeError(
                "root.state.native_state is None, but use_native=True. "
                "Ensure Gomoku(use_native=True) was used to create states."
            )

        root_native_state = root.state.native_state
        noise_pending = bool(add_noise and not getattr(root, "noise_applied", False))
        batch_size = engine.params.batch_infer_size

        if batch_size > 1:
            if hasattr(engine.inference, "infer_async"):
                self._run_async_mcts(engine, root, root_native_state, sims, batch_size, noise_pending)
            else:
                self._run_sync_batch_mcts(engine, root, root_native_state, sims, batch_size, noise_pending)
        else:
            self._run_encoded_mcts(engine, root, root_native_state, sims, noise_pending)

    def _run_async_mcts(self, engine: 'SequentialEngine', root, root_native_state, sims, batch_size, noise_pending):
        import ray

        inflight_refs = {}  # handle -> ray.ObjectRef
        next_handle = 0

        def async_dispatcher(py_batch: np.ndarray) -> int:
            nonlocal next_handle
            # Convert to tensor and dispatch asynchronously (non-blocking)
            tensor = torch.from_numpy(py_batch).to(device=engine._inference_device, dtype=torch.float32)
            ref = engine.inference.infer_async(tensor)

            h = next_handle
            next_handle += 1
            inflight_refs[h] = ref
            return h

        def async_checker(handles: List[int], timeout_s: float) -> List[Tuple[int, Tuple[np.ndarray, np.ndarray]]]:
            # Filter refs that C++ is interested in (passed via handles)
            target_refs = [inflight_refs[h] for h in handles if h in inflight_refs]
            if not target_refs:
                return []

            # Wait for any of the target refs to be ready
            ready_refs, _ = ray.wait(
                target_refs,
                num_returns=len(target_refs),
                timeout=timeout_s,
                fetch_local=True
            )

            results = []
            for ref in ready_refs:
                # Find the handle corresponding to this ref
                # (Simple linear search is fine as handles list is small, typically inflight_limit)
                h_found = None
                for h in handles:
                    if inflight_refs.get(h) == ref:
                        h_found = h
                        break

                if h_found is not None:
                    try:
                        policy_t, value_t = ray.get(ref)
                        policy_np = policy_t.detach().cpu().numpy()
                        values_np = value_t.detach().cpu().numpy().reshape(-1, 1)
                        results.append((h_found, (policy_np, values_np)))
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).error(f"Async inference error for handle {h_found}: {e}")

                    # Remove from tracking
                    del inflight_refs[h_found]

            return results

        try:
            visits = engine._native_engine.run_mcts_async_encoded(
                root_native_state,
                sims,
                batch_size,
                async_dispatcher,
                async_checker
            )
            self._sync_visits(engine, root, visits)
        finally:
            # Cleanup any lingering refs
            for ref in inflight_refs.values():
                try:
                    ray.cancel(ref)
                except Exception:
                    pass
            inflight_refs.clear()

    def _run_sync_batch_mcts(self, engine: 'SequentialEngine', root, root_native_state, sims, batch_size, noise_pending):
        internal_noise_pending = noise_pending

        def native_evaluator_batch_encoded(py_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            nonlocal internal_noise_pending
            tensor = torch.from_numpy(py_batch).to(device=engine._inference_device, dtype=torch.float32)
            logits, values = engine.inference.infer(tensor)
            if internal_noise_pending:
                internal_noise_pending = False
                root.noise_applied = True
            logits_np = logits.detach().cpu().numpy()
            values_np = values.detach().cpu().numpy().reshape(-1, 1)
            return logits_np, values_np

        visits = engine._native_engine.run_mcts_batch_encoded(
            root_native_state,
            sims,
            batch_size,
            native_evaluator_batch_encoded,
        )
        self._sync_visits(engine, root, visits)

    def _run_encoded_mcts(self, engine: 'SequentialEngine', root, root_native_state, sims, noise_pending):
        internal_noise_pending = noise_pending

        def native_evaluator_encoded(features: np.ndarray, legal_indices: np.ndarray) -> Tuple[List[float], float]:
            nonlocal internal_noise_pending
            tensor = torch.from_numpy(features).unsqueeze(0).to(device=engine._inference_device, dtype=torch.float32)
            logits, value_tensor = engine.inference.infer(tensor)
            if logits.dim() == 1: logits = logits.unsqueeze(0)
            legal_mask = torch.zeros(engine.game.action_size, dtype=torch.bool, device=logits.device)
            if legal_indices.size > 0:
                idx_t = torch.from_numpy(legal_indices).to(device=logits.device, dtype=torch.long)
                legal_mask.index_fill_(0, idx_t, True)
            apply_noise = False
            if internal_noise_pending:
                apply_noise = True
                internal_noise_pending = False
                root.noise_applied = True
            policy = engine._masked_policy_from_logits(logits, legal_mask, apply_dirichlet=apply_noise)
            if policy.dim() > 1: policy = policy.squeeze(0)

            # C++ expects LOGITS (it applies softmax). We computed PROBS (policy).
            # To pass PROBS correctly (preserving Noise/Masking), we return LOG(PROBS).
            # C++: Softmax(Log(P)) = P.
            pol_np = policy.detach().cpu().numpy()

            # Avoid log(0)
            epsilon = 1e-30
            log_policy = np.log(np.maximum(pol_np, epsilon))

            value = float(value_tensor.squeeze().item())
            return log_policy.tolist(), value

        visits = engine._native_engine.run_mcts_encoded(
            root_native_state,
            sims,
            native_evaluator_encoded,
        )
        self._sync_visits(engine, root, visits)

    def _sync_visits(self, engine: 'SequentialEngine', root: 'TreeNode', visits: List[Tuple[int, int]]) -> None:
        """Sync C++ visits to Python TreeNode."""
        total_visits = sum(int(count) for _, count in visits)
        root.visit_count = total_visits
        root.children = {}
        denom = max(total_visits, 1)
        for move_idx, count in visits:
            if move_idx < 0 or move_idx >= engine.game.action_size:
                continue
            action = (move_idx % engine.game.col_count, move_idx // engine.game.col_count)
            next_state = engine.game.get_next_state(root.state, action, root.state.next_player)
            from gomoku.pvmcts.treenode import TreeNode
            child = TreeNode(
                state=next_state,
                parent=root,
                action_taken=action,
                prior=float(count) / float(denom),
            )
            child.visit_count = int(count)
            root.children[action] = child

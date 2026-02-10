from collections import Counter
import os
import time

import numpy as np
import torch

from gomoku.core.gomoku import Gomoku
from gomoku.inference.ray_client import RayInferenceClient
from gomoku.pvmcts.search.engine import SearchEngine
from gomoku.pvmcts.search.ray.batch_inference_manager import (
    BatchInferenceManager,
    PendingNodeInfo,
)
from gomoku.pvmcts.treenode import TreeNode
from gomoku.utils.config.loader import MctsConfig


class RayAsyncEngine(SearchEngine):
    """
    Asynchronous MCTS engine that pipelines selection and remote inference over Ray.

    Notes
    -----
    - Aggressively dispatches ready batches to keep the pipeline busy.
    - Supports tree reuse by injecting Dirichlet noise into already expanded roots.
    - Applies simple retry by reselecting when a leaf is already in flight.
    - Tracks virtual loss with cleanup guards to prevent negative pending counts.
    """

    def __init__(
        self,
        game: Gomoku,
        params: MctsConfig,
        inference_client: RayInferenceClient,
        batch_size: int | None = None,
        min_batch_size: int | None = None,
        max_wait_ms: int | None = None,
        async_inflight_limit: int | None = None,
        use_fp16: bool = True,
    ):
        """
        Initialize the Ray-based asynchronous search engine.

        Parameters
        ----------
        game : Gomoku
            Game environment providing state transitions and encoding.
        params : MctsConfig
            MCTS configuration (search budget, exploration constants, noise params).
        inference_client : RayInferenceClient
            Asynchronous inference client backed by a Ray actor.
        batch_size : int or None, optional
            Preferred batch size for inference
        min_batch_size : int or None, optional
            Minimum batch size before dispatch
        max_wait_ms : int or None, optional
            Maximum wait time in milliseconds before dispatching a partial batch
        async_inflight_limit : int or None, optional
            Maximum number of concurrent in-flight batches
        use_fp16 : bool, optional
            Whether FP16 is preferred for inference (passed through to the base engine).
        """
        super().__init__(
            game,
            params,
            inference_client,
            batch_size,
            min_batch_size,
            max_wait_ms,
            async_inflight_limit,
            use_fp16,
        )

        self.pid = os.getpid()

        self.manager = BatchInferenceManager(
            client=inference_client,
            batch_size=self.batch_size,
            max_wait_ms=self.max_wait_ms,
            max_inflight_batches=self.async_inflight_limit,
        )
        self._seed_random_generators()

    def search(self, roots: TreeNode | list[TreeNode], add_noise: bool = False) -> None:
        """
        Run asynchronous MCTS with batched remote inference.

        Parameters
        ----------
        roots : TreeNode or list[TreeNode]
            One or more start nodes to search; supports tree reuse.
        add_noise : bool
            Whether to apply Dirichlet noise on root reuse and first expansions.

        Returns
        -------
        None
            Mutates the provided root nodes in-place with visits, priors, and values.

        Notes
        -----
        - Applies Dirichlet noise to reused roots and on first-time expansions.
        - Uses virtual loss during selection to discourage duplicate in-flight leaves.
        - Dispatches batches as soon as size/timeout conditions are met to keep GPUs busy.
        """
        if isinstance(roots, TreeNode):
            roots = [roots]
        else:
            roots = list(roots or [])

        roots = [r for r in roots if r is not None]
        if not roots:
            return

        if add_noise and self.params.dirichlet_epsilon > 0:
            for root in roots:
                if not root.is_leaf:
                    self._ensure_root_noise(root)

        target_visits = int(self.params.num_searches)
        finished_roots: set[TreeNode] = set()

        pending_by_root: Counter[TreeNode] = Counter()
        inflight_paths: dict[TreeNode, list[TreeNode]] = {}
        inflight_to_root: dict[TreeNode, TreeNode] = {}

        device = torch.device("cpu")
        dtype = torch.float32

        inflight_limit = self.async_inflight_limit
        batch_size = self.batch_size
        # inflight_limit가 지정되지 않았다면 최소한 루트 수만큼의 배치까지만 허용해
        # 무한대 누적을 방지한다.
        effective_limit = (
            inflight_limit if inflight_limit is not None else max(1, len(roots))
        )
        max_pending = effective_limit * batch_size

        try:
            while len(finished_roots) < len(roots):
                self._selection_phase(
                    roots,
                    finished_roots,
                    pending_by_root,
                    inflight_to_root,
                    inflight_paths,
                    max_pending,
                    target_visits,
                    dtype,
                    device,
                )

                self._dispatch_phase()

                self._drain_phase(
                    finished_roots,
                    pending_by_root,
                    inflight_to_root,
                    inflight_paths,
                    target_visits,
                    dtype,
                    device,
                    add_noise,
                )
        finally:
            # If this search exits early (errors, interrupts), ensure the tree does not
            # keep stale "in-flight" markers that would poison future searches.
            for path in inflight_paths.values():
                for node in path:
                    node.pending_visits = max(0, node.pending_visits - 1)

            inflight_paths.clear()
            inflight_to_root.clear()
            pending_by_root.clear()

            # Drain any remaining in-flight results and clear pending queue.
            self.manager.cleanup()

    def _selection_phase(
        self,
        roots: list[TreeNode],
        finished_roots: set[TreeNode],
        pending_by_root: Counter[TreeNode],
        inflight_to_root: dict[TreeNode, TreeNode],
        inflight_paths: dict[TreeNode, list[TreeNode]],
        max_pending: int,
        target_visits: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Selection + enqueue with virtual loss application."""
        for root in roots:
            if root in finished_roots:
                continue

            current_pending = pending_by_root[root]
            if root.visit_count + current_pending >= target_visits:
                if current_pending == 0:
                    pending_by_root.pop(root, None)
                    finished_roots.add(root)
                continue

            if self.manager.pending_count() >= max_pending:
                break

            leaf, path = self._select_path(root)
            if leaf in inflight_to_root:
                continue

            value, is_terminal = self._evaluate_terminal(leaf)
            if is_terminal:
                leaf.backup(value)
                if leaf is root:
                    finished_roots.add(root)
                continue

            # Virtual loss apply with rollback guard
            try:
                for node in path:
                    node.pending_visits += 1
                inflight_to_root[leaf] = root
                inflight_paths[leaf] = path
                pending_by_root[root] += 1

                native_payload = None
                if getattr(self.game, "use_native", False) and leaf.state.native_state is not None:
                    features_np = np.array(
                        self.game._native_core.write_state_features(  # noqa: SLF001
                            leaf.state.native_state
                        ),
                        copy=False,
                        dtype=np.float32,
                    )
                    state_tensor = torch.as_tensor(features_np, dtype=dtype, device=device)
                    native_payload = features_np
                else:
                    state_tensor = self._encode_state_to_tensor(
                        leaf.state, dtype=dtype, device=device
                    )
                is_start_node = leaf is root
                self.manager.enqueue(
                    PendingNodeInfo(node=leaf, is_start_node=is_start_node),
                    state_tensor,
                    native_state=native_payload,
                )
            except Exception:
                inflight_to_root.pop(leaf, None)
                inflight_paths.pop(leaf, None)
                if pending_by_root[root] > 0:
                    pending_by_root[root] -= 1
                for node in path:
                    node.pending_visits = max(0, node.pending_visits - 1)
                raise

    def _dispatch_phase(self) -> None:
        """Dispatch ready batches until inflight limit blocks."""
        while self.manager.dispatch_ready():
            pass

    def _drain_phase(
        self,
        finished_roots: set[TreeNode],
        pending_by_root: Counter[TreeNode],
        inflight_to_root: dict[TreeNode, TreeNode],
        inflight_paths: dict[TreeNode, list[TreeNode]],
        target_visits: int,
        dtype: torch.dtype,
        device: torch.device,
        add_noise: bool,
    ) -> None:
        """Drain results and apply expansion/backup."""
        results = self.manager.drain_ready(timeout_s=0.001)
        if not results:
            if not inflight_to_root and finished_roots:
                return
            if not inflight_to_root and target_visits > 0:
                time.sleep(0.001)
            return

        for res in results:
            batch_policy = torch.as_tensor(
                res.policy_logits, dtype=dtype, device=device
            )
            batch_values = torch.as_tensor(res.values, dtype=dtype, device=device).view(
                -1
            )

            if batch_policy.size(0) != len(res.mapping):
                raise RuntimeError(
                    f"Batch mismatch! Expected {len(res.mapping)}, "
                    f"got {batch_policy.size(0)}"
                )

            for i, mapping in enumerate(res.mapping):
                leaf = mapping.node
                root = inflight_to_root.get(leaf)
                path = inflight_paths.get(leaf)
                if root is None or path is None:
                    continue

                try:
                    if root in finished_roots:
                        continue
                    self._expand_and_backup(
                        leaf,
                        batch_policy[i],
                        float(batch_values[i].item()),
                        is_start_node=mapping.is_start_node,
                        add_noise=add_noise,
                    )
                finally:
                    inflight_to_root.pop(leaf, None)
                    inflight_paths.pop(leaf, None)

                    for node in path:
                        node.pending_visits = max(0, node.pending_visits - 1)

                    pending_by_root[root] -= 1
                    if pending_by_root[root] <= 0:
                        pending_by_root.pop(root, None)
                    if (
                        root.visit_count >= target_visits
                        and root not in finished_roots
                        and pending_by_root.get(root, 0) == 0
                    ):
                        finished_roots.add(root)

    def _select_path(self, root: TreeNode) -> tuple[TreeNode, list[TreeNode]]:
        """Select a path using UCB and return the reached leaf with its path."""
        node = root
        path = [node]
        while not node.is_leaf:
            node = node.best_child(self.params.C)
            path.append(node)
        return node, path

    def _expand_and_backup(
        self,
        node: TreeNode,
        logits: torch.Tensor,
        value: float,
        is_start_node: bool,
        add_noise: bool,
    ) -> None:
        """Expand a leaf with logits and backpropagate the resulting value."""
        legal_mask = self._legal_mask_tensor(node.state, logits.device)
        policy = self._masked_policy_from_logits(
            logits, legal_mask, apply_dirichlet=is_start_node and add_noise
        )
        self._expand_with_policy(node, policy)
        self._backup(node, value)

    def __repr__(self) -> str:
        return f"<RayAsyncEngine(pid={self.pid})>"

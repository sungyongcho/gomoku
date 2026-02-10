import torch

from gomoku.pvmcts.search.engine import SearchEngine
from gomoku.pvmcts.treenode import TreeNode


class VectorizeEngine(SearchEngine):
    """
    CPU-only batched search that keeps control flow minimal.

    Notes
    -----
        - Maintains an active set to skip finished games.
        - Respects configured ``num_searches`` by skipping start nodes that already
          reached the target visit count.
        - Applies Dirichlet noise only on the initial start-node expansion.
    """

    def __init__(
        self,
        game,
        mcts_params,
        inference_client,
        batch_size: int | None = None,
        min_batch_size: int | None = None,
        max_wait_ms: int | None = None,
        async_inflight_limit: int | None = None,
        use_fp16: bool = True,
    ):
        super().__init__(
            game,
            mcts_params,
            inference_client,
            batch_size=batch_size,
            min_batch_size=min_batch_size,
            max_wait_ms=max_wait_ms,
            async_inflight_limit=async_inflight_limit,
            use_fp16=use_fp16,
        )

    def search(self, roots: TreeNode | list[TreeNode], add_noise: bool = False) -> None:
        """
        Run batched MCTS simulations on CPU.

        Parameters
        ----------
        roots : TreeNode or list[TreeNode]
            Starting node(s) to search.
        add_noise : bool
            Whether to apply Dirichlet noise on initial start-node expansion/reuse.

        Notes
        -----
        - Unvisited start nodes are expanded once with Dirichlet noise.
        - Simulations proceed until each active start node reaches ``num_searches``.
        - Newly encountered leaves are grouped for batch inference.
        """
        if isinstance(roots, TreeNode):
            roots = [roots]
        if not roots:
            return

        target_searches = int(self.params.num_searches)
        if target_searches <= 0:
            return

        done_nodes: set[TreeNode] = set()

        unexpanded = []
        for start_node in roots:
            if start_node is None:
                done_nodes.add(start_node)
                continue
            if add_noise and start_node.children:
                self._ensure_root_noise(start_node)
            if start_node.visit_count == 0:
                value, is_terminal = self._evaluate_terminal(start_node)
                if is_terminal:
                    self._backup(start_node, value)
                    done_nodes.add(start_node)
                else:
                    unexpanded.append(start_node)

        if unexpanded:
            self._process_batch(
                unexpanded,
                is_start_node_batch=True,
                add_noise=add_noise,
            )

        while True:
            active_nodes = [
                node
                for node in roots
                if node not in done_nodes and node.visit_count < target_searches
            ]

            if not active_nodes:
                break

            leaf_nodes: list[TreeNode] = []

            for start_node in active_nodes:
                node = start_node

                # Selection
                while not node.is_leaf:
                    node = node.best_child(self.params.C)

                # Evaluation (Terminal Check)
                value, is_terminal = self._evaluate_terminal(node)

                if is_terminal:
                    self._backup(node, value)
                    if node is start_node:
                        done_nodes.add(start_node)
                else:
                    leaf_nodes.append(node)

            if leaf_nodes:
                self._process_batch(
                    leaf_nodes,
                    is_start_node_batch=False,
                    add_noise=add_noise,
                )

    def _process_batch(
        self, nodes: list[TreeNode], is_start_node_batch: bool, add_noise: bool
    ) -> None:
        inference_device = getattr(self.inference, "device", None)
        if isinstance(inference_device, torch.device):
            device = inference_device
        elif isinstance(inference_device, str):
            device = torch.device(inference_device)
        else:
            device = torch.device("cpu")

        dtype = (
            torch.float16
            if (self.use_fp16 and device.type == "cuda")
            else torch.float32
        )

        batch_tensors = [
            self._encode_state_to_tensor(n.state, dtype=dtype, device=device)
            for n in nodes
        ]
        batch_input = torch.stack(batch_tensors)

        if hasattr(self.inference, "infer_batch"):
            policy_logits, values = self.inference.infer_batch(batch_input)
        else:
            policy_logits, values = self.inference.infer(batch_input)

        if policy_logits.dim() == 1:
            policy_logits = policy_logits.unsqueeze(0)
        values_flat = values.view(-1)

        for i, node in enumerate(nodes):
            val_idx = i if i < values_flat.numel() else -1
            value = float(values_flat[val_idx].item())
            logits = policy_logits[i]

            legal_mask = self._legal_mask_tensor(node.state, logits.device)
            policy = self._masked_policy_from_logits(
                logits,
                legal_mask,
                # True only for initial start-node expansion
                apply_dirichlet=is_start_node_batch and add_noise,
            )

            self._expand_with_policy(node, policy)
            self._backup(node, value)

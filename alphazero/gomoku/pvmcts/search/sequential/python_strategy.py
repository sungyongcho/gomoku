import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from gomoku.pvmcts.search.sequential.engine import SequentialEngine
    from gomoku.pvmcts.treenode import TreeNode

class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    @abstractmethod
    def search(self, engine: 'SequentialEngine', roots: List['TreeNode'], add_noise: bool) -> None:
        """Execute the search algorithm."""
        pass

class PythonSearchStrategy(SearchStrategy):
    """Pure Python implementation of sequential MCTS."""

    def search(self, engine: 'SequentialEngine', roots: List['TreeNode'], add_noise: bool) -> None:
        for root in roots:
            if root is None:
                continue
            self._search_single(engine, root, add_noise)

    def _search_single(
        self,
        engine: 'SequentialEngine',
        root: 'TreeNode',
        add_noise: bool,
    ) -> None:
        """Execute search for a single root."""
        num_searches = int(engine.params.num_searches)
        dtype = torch.float32
        device = torch.device("cpu")

        # Initial expansion or reused root handling
        if root.visit_count == 0:
            value_term, is_term = engine._evaluate_terminal(root)
            if is_term:
                root.backup(value_term)
                return

            logits, value = engine._infer_single(
                node_state=root.state,
                dtype=dtype,
                device=device,
                apply_dirichlet=add_noise,
            )

            engine._expand_with_policy(root, logits)
            engine._backup(root, value)
        else:
            if add_noise and root.children:
                engine._ensure_root_noise(root)

        # [Simulation Loop]
        for _ in range(max(0, num_searches - root.visit_count)):
            node = root
            # Selection
            while not node.is_leaf:
                node = node.best_child(engine.params.C)

            # Evaluation (Terminal Check)
            value, is_term = engine._evaluate_terminal(node)
            if is_term:
                engine._backup(node, value)
                continue

            # Expansion (Non-terminal Leaf)
            if node.visit_count == 0:
                logits, value = engine._infer_single(
                    node_state=node.state,
                    dtype=dtype,
                    device=device,
                    apply_dirichlet=False,
                )
                engine._expand_with_policy(node, logits)
            else:
                value = 0.0

            engine._backup(node, value)

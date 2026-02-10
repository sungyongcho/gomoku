import math
from typing import Optional

import numpy as np

from gomoku.core.gomoku import GameState, Gomoku


class TreeNode:
    """Lightweight MCTS node (kept simple for portability)."""

    __slots__ = (
        "state",
        "parent",
        "action_taken",
        "prior",
        "children",
        "visit_count",
        "value_sum",
        "pending_visits",
        "noise_applied",
        "native_node",
        "_terminal_cache",
        "_legal_indices_cache",
    )

    def __init__(
        self,
        state: GameState,
        parent: Optional["TreeNode"] = None,
        action_taken: tuple[int, int] | None = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children: dict[tuple[int, int], TreeNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.pending_visits: int = 0
        self.noise_applied: bool = False
        self.native_node = None
        self._terminal_cache: tuple[float, bool] | None = None
        self._legal_indices_cache: np.ndarray | None = None

    def get_terminal_info(self, game: Gomoku) -> tuple[float, bool]:
        """Return cached terminal info or calculate it."""
        if self._terminal_cache is None:
            if self.action_taken is None:
                # Root node or special case
                if self.state.last_move_idx != -1 and game.check_win(
                    self.state, self.state.last_move_idx
                ):
                    self._terminal_cache = (-1.0, True)
                elif self.state.empty_count <= 0:
                    self._terminal_cache = (0.0, True)
                else:
                    self._terminal_cache = (0.0, False)
            else:
                # Child node: check from parent's perspective using action_taken
                raw_value, terminated = game.get_value_and_terminated(
                    self.state, self.action_taken
                )
                if terminated:
                    # Value is for the winner (previous player).
                    # Current node perspective (next player) is -value.
                    self._terminal_cache = (-float(raw_value), True)
                else:
                    self._terminal_cache = (float(raw_value), False)
        return self._terminal_cache

    def get_legal_indices(self, game: Gomoku) -> np.ndarray:
        """Return cached legal indices."""
        if self._legal_indices_cache is None:
            # Leverage state cache if available
            if self.state.legal_indices_cache is not None:
                self._legal_indices_cache = self.state.legal_indices_cache
            else:
                # Calculate and cache
                game.get_legal_moves(self.state)  # Populates state.legal_indices_cache
                self._legal_indices_cache = self.state.legal_indices_cache

            if self._legal_indices_cache is None:
                self._legal_indices_cache = np.array([], dtype=np.int64)

        return self._legal_indices_cache

    @property
    def is_leaf(self) -> bool:
        """Return True if the node has no children."""
        return not self.children

    @property
    def q_value(self) -> float:
        """Average value (Q); returns 0.0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def get_ucb(self, child: "TreeNode", c_puct: float) -> float:
        """
        Compute UCB score (parent perspective).

        U = c_puct * prior * sqrt(N_parent) / (1 + N_child)

        Parameters
        ----------
        child : TreeNode
            Child node to evaluate.
        c_puct : float
            Exploration constant.

        Returns
        -------
        float
            UCB score for the child from the parent's perspective.
        """
        q_value = -child.q_value

        parent_n = self.visit_count + self.pending_visits
        child_n = child.visit_count + child.pending_visits

        parent_sqrt = math.sqrt(max(1, parent_n))

        u_value = c_puct * child.prior * (parent_sqrt / (1 + child_n))
        return q_value + u_value

    def best_child(self, c_puct: float) -> "TreeNode":
        """
        Return child with the maximum UCB score.

        Parameters
        ----------
        c_puct : float
            Exploration constant.

        Returns
        -------
        TreeNode
            Child node with the highest UCB score.

        Raises
        ------
        ValueError
            If called on a leaf node.
        """
        if not self.children:
            raise ValueError(
                "Attempted to select child from a leaf node. "
                f"Action: {self.action_taken}"
            )
        # Inline UCB to compute parent_sqrt once instead of per-child.
        parent_n = self.visit_count + self.pending_visits
        parent_sqrt = math.sqrt(max(1, parent_n))
        best = None
        best_score = -math.inf
        for child in self.children.values():
            child_n = child.visit_count + child.pending_visits
            score = -child.q_value + c_puct * child.prior * (parent_sqrt / (1 + child_n))
            if score > best_score:
                best_score = score
                best = child
        return best

    def expand(
        self, policy: np.ndarray, legal_moves_mask: np.ndarray, game: Gomoku
    ) -> None:
        """
        Expand children using the given policy and legal mask.

        Parameters
        ----------
        policy : np.ndarray
            Policy probabilities over flat action space.
        legal_moves_mask : np.ndarray
            Boolean mask of legal actions (same length as policy).
        game : Gomoku
            Game instance for state transitions.
        """
        if not self.is_leaf:
            return

        masked_policy = policy * legal_moves_mask
        policy_sum = masked_policy.sum()

        if policy_sum > 1e-6:
            final_policy = masked_policy / policy_sum
        else:
            legal_count = legal_moves_mask.sum()
            if legal_count > 0:
                final_policy = legal_moves_mask / legal_count
            else:
                return

        valid_indices = np.nonzero(final_policy)[0]
        col_count = game.col_count

        next_player = self.state.next_player

        for idx in valid_indices:
            prob = float(final_policy[idx])
            x = idx % col_count
            y = idx // col_count
            action = (x, y)

            if action in self.children:
                continue

            next_state = game.get_next_state(self.state, action, next_player)

            self.children[action] = TreeNode(
                state=next_state, parent=self, action_taken=action, prior=prob
            )

    def backup(self, value_for_this_node: float) -> None:
        """
        Propagate a leaf value to the root (node owner perspective).

        Parameters
        ----------
        value_for_this_node : float
            Value from this node's current player perspective.
        """
        node = self
        current_perspective_value = value_for_this_node

        while node is not None:
            node.visit_count += 1
            node.value_sum += current_perspective_value

            # Parent is opponent; flip perspective on the way up.
            current_perspective_value = -current_perspective_value
            node = node.parent

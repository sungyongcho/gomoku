from __future__ import annotations

import copy
import math
import random

import numpy as np

from gomoku import GameState, Gomoku, convert_coordinates_to_index


class Node:
    def __init__(
        self, game: Gomoku, args, state: GameState, parent=None, action_taken=None
    ):
        self.game = game
        self.args = args
        self.state = state
        self.parent: Node | None = parent
        self.action_taken = action_taken

        self.children = []

        self.expandable_moves = [
            idx  # ① the value you keep
            for idx in (  # ② loop variable
                convert_coordinates_to_index(c) for c in game.get_legal_moves(state)
            )
            if idx is not None  # ③ filter
        ]
        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child: Node):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args["C"] * math.sqrt(
            (math.log(self.visit_count) / child.visit_count)
        )

    def expand(self):
        action = random.choice(self.expandable_moves)
        self.expandable_moves.remove(action)

        child_state: GameState = copy.deepcopy(self.state)
        child_state = self.game.get_next_state(
            child_state, action, child_state.next_player
        )

        child: Node = Node(self.game, self.args, child_state, self, action)

        self.children.append(child)
        return child

    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(
            self.state, self.action_taken
        )
        # *** here also how are you sure that the node will be same or different from the root node's player? ***
        value = -value

        if is_terminal:
            return value

        rollout_state: GameState = copy.deepcopy(self.state)
        rollout_player = rollout_state.next_player
        while True:
            valid_moves = [
                idx  # ① the value you keep
                for idx in (  # ② loop variable
                    convert_coordinates_to_index(c)
                    for c in self.game.get_legal_moves(rollout_state)
                )
                if idx is not None  # ③ filter
            ]
            action = random.choice(valid_moves)
            rollout_state = self.game.get_next_state(
                rollout_state, action, rollout_player
            )
            value, is_terminal = self.game.get_value_and_terminated(
                rollout_state, action
            )
            if is_terminal:
                if rollout_player != self.state.next_player:
                    value = -value
                return value

            rollout_player = rollout_state.next_player

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        # *** here also how are you sure that the node will be same or different from the root node's player? ***
        value = -value
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game: Gomoku, args):
        self.game = game
        self.args = args

    def search(self, state):
        # 1. begin search by selecting the root node
        root = Node(self.game, self.args, state)

        # 2. searching by using number of 'num_search' args
        for search in range(self.args["num_searches"]):
            # 3. select the node to traverse down, which will be the root
            node: Node = root

            # 4. selecting by traversing down based on getting the best ucb value. it will traverse down until it reaches leaf node
            #    which means from root, it will stay at root, but if child exist, it will select based on the best ucb value among all the child nodes
            while node.is_fully_expanded():
                node = node.select()

            # 5. get the value and terminal state of the game of the selected node's state.
            value, is_terminal = self.game.get_value_and_terminated(
                node.state, node.action_taken
            )
            # 6. reverse the value.
            # *** but what if the node's player is same as the player? ***
            value = -value

            # 7. if the current node's state is not terminal, which means there is a chance that the game can be continued
            #    so, it will expand the game and simulate the game.
            if not is_terminal:
                # expansion
                node = node.expand()  # 8. current node will be changed to the expanded node, because later it will be backpropagated based on the leaf of selected
                # simulation
                value = node.simulate()  # 9. simulate the game

            # backpropagation
            node.backpropagate(
                value
            )  # 10. with selected node result, it will backpropagate the tree of the result of value

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            x, y = child.action_taken
            idx = x + y * self.game.col_count
            action_probs[idx] = child.visit_count
        action_probs /= np.sum(action_probs)
        # print(action_probs)
        return action_probs

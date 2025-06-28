from __future__ import annotations

import copy
import math

import numpy as np
import torch

from gomoku import GameState, Gomoku, convert_coordinates_to_index


class Node:
    def __init__(
        self,
        game: Gomoku,
        args,
        state: GameState,
        parent=None,
        action_taken=None,
        prior=0,
    ):
        self.game = game
        self.args = args
        self.state = state
        self.parent: Node | None = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = []

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

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
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return (
            q_value
            + self.args["C"]
            * (math.sqrt(self.visit_count) / (child.visit_count + 1))
            * child.prior
        )

    def expand(self, policy):
        for idx, prob in enumerate(policy):
            if prob > 0:
                x = idx % self.game.col_count
                y = idx // self.game.col_count
                action = (x, y)
                child_state: GameState = copy.deepcopy(self.state)
                child_state = self.game.get_next_state(
                    child_state, action, child_state.next_player
                )

                child: Node = Node(
                    self.game, self.args, child_state, self, action, prob
                )

                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = -value
        if self.parent is not None:
            self.parent.backpropagate(value)


class PVMCTS:
    def __init__(self, game: Gomoku, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state)

        for search in range(self.args["num_searches"]):
            node: Node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(
                node.state, node.action_taken
            )
            value = -value

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state)).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

                legal_mask = np.zeros(self.game.action_size, np.float32)
                for coord in self.game.get_legal_moves(node.state):
                    x, y = convert_coordinates_to_index(coord)
                    legal_mask[x + y * self.game.col_count] = 1.0

                policy *= legal_mask
                policy /= np.sum(policy)
                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            x, y = child.action_taken
            idx = x + y * self.game.col_count
            action_probs[idx] = child.visit_count
        action_probs /= np.sum(action_probs)
        print(action_probs)
        return action_probs

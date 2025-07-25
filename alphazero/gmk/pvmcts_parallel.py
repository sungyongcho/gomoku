from __future__ import annotations

import math

import numpy as np
import torch

from gomoku import GameState, Gomoku, convert_coordinates_to_index
from policy_value_net import PolicyValueNet


class Node:
    def __init__(
        self,
        game: Gomoku,
        args,
        state: GameState,
        parent=None,
        action_taken=None,
        prior=0,
        visit_count=0,
    ):
        self.game = game
        self.args = args
        self.state = state
        self.parent: Node | None = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = []

        self.visit_count = visit_count
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
        q = (
            0
            if child.visit_count == 0
            else 1
            - ((child.value_sum / child.visit_count) + 1)
            / 2  # normalize + switch to parent's perspective
        )
        return (
            q
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

                child_state = self.game.get_next_state(
                    self.state, action, self.state.next_player
                )
                child: Node = Node(
                    self.game, self.args, child_state, self, action, prob
                )

                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = -value  # parent -> opposite player
        if self.parent is not None:
            self.parent.backpropagate(value)


class PVMCTSParallel:
    def __init__(self, game: Gomoku, args, model):
        self.game = game
        self.args = args
        self.model: PolicyValueNet = model

    @torch.no_grad()
    def search(self, states, spGames):
        policy, _ = self.model(
            torch.tensor(
                self.game.get_encoded_state(states),
                device=self.model.device,
            )
        )
        policy = torch.softmax(policy, axis=1).detach().cpu().numpy()
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args[
            "dirichlet_epsilon"
        ] * np.random.dirichlet(
            [self.args["dirichlet_alpha"]] * self.game.action_size, size=policy.shape[0]
        )

        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            legal_mask = np.zeros(self.game.action_size, np.float32)
            for coord in self.game.get_legal_moves(states[i]):
                x, y = convert_coordinates_to_index(coord)
                legal_mask[x + y * self.game.col_count] = 1.0
            spg_policy *= legal_mask
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visit_count=1)

            spg.root.expand(spg_policy)

        for search in range(self.args["num_searches"]):
            for spg in spGames:
                spg.node = None
                node: Node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(
                    node.state, node.action_taken
                )
                value = -value  # important

                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node

            expandable_spGames = [
                mappingIdx
                for mappingIdx in range(len(spGames))
                if spGames[mappingIdx].node is not None
            ]

            if len(expandable_spGames) > 0:
                states = np.stack(
                    [
                        spGames[mappingIdx].node.state
                        for mappingIdx in expandable_spGames
                    ]
                )
                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(states),
                        device=self.model.device,
                    )
                )
                policy = torch.softmax(policy, axis=1).detach().cpu().numpy()
                value = value.cpu().numpy()

            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node

                spg_policy, spg_value = policy[i], value[i]

                legal_mask = np.zeros(self.game.action_size, np.float32)
                for coord in self.game.get_legal_moves(node.state):
                    x, y = convert_coordinates_to_index(coord)
                    legal_mask[x + y * self.game.col_count] = 1.0

                spg_policy *= legal_mask
                spg_policy = np.maximum(spg_policy, 1e-8)
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)

                node.backpropagate(spg_value)

        # action_probs = np.zeros(self.game.action_size)
        # for child in root.children:
        #     x, y = child.action_taken
        #     idx = x + y * self.game.col_count
        #     action_probs[idx] = child.visit_count
        # action_probs /= np.sum(action_probs)
        # # print(action_probs)
        # return action_probs

from __future__ import annotations

import math

import numpy as np
import torch
from tictactoe import TicTacToe

torch.manual_seed(0)


class Node:
    def __init__(
        self,
        game: TicTacToe,
        args,
        state,
        parent: Node = None,
        action_taken=None,
        prior=0,
    ):
        self.game = game
        self.args = args
        self.state = state
        self.parent: Node = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []
        # self.expandable_moves = game.get_valid_moves(state)

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        # return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
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

    # def get_ucb(self, child: Node):
    #     q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
    #     return q_value + self.args["C"] * math.sqrt(
    #         math.log(self.visit_count) / child.visit_count
    #     )

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

    # def expand(self):
    #     action = np.random.choice(np.where(self.expandable_moves == 1)[0])
    #     self.expandable_moves[action] = 0

    #     child_state = self.state.copy()
    #     child_state = self.game.get_next_state(child_state, action, 1)
    #     child_state = self.game.change_perspective(child_state, player=-1)

    #     child = Node(self.game, self.args, child_state, self, action)
    #     self.children.append(child)
    #     return child

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

    # def simulate(self):
    #     value, is_terminal = self.game.get_value_and_terminated(
    #         self.state, self.action_taken
    #     )
    #     value = self.game.get_opponent_value(value)

    #     if is_terminal:
    #         return value

    #     rollout_state = self.state.copy()
    #     rollout_player = 1
    #     while True:
    #         valid_moves = self.game.get_valid_moves(rollout_state)
    #         action = np.random.choice(np.where(valid_moves == 1)[0])
    #         rollout_state = self.game.get_next_state(
    #             rollout_state, action, rollout_player
    #         )
    #         value, is_terminal = self.game.get_value_and_terminated(
    #             rollout_state, action
    #         )
    #         if is_terminal:
    #             if rollout_player == -1:
    #                 value = self.game.get_opponent_value(value)
    #             return value

    #         rollout_player = self.game.get_opponent(rollout_player)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game: TicTacToe, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        # define root
        root = Node(self.game, self.args, state)

        for search in range(self.args["num_searches"]):
            # selection
            node = root
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(
                node.state, node.action_taken
            )
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state)).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()
                node.expand(policy)
                # # expansion
                # node = node.expand()
                # # simulation
                # value = node.simulate()

            # backpropagation
            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


# tictactoe = TicTacToe()

# player = 1

# args = {"C": 1.41, "num_searches": 1000}
# mcts = MCTS(tictactoe, args)
# state = tictactoe.get_initial_state()

# while True:
#     print(state)
#     if player == 1:
#         valid_moves = tictactoe.get_valid_moves(state)
#         print(
#             "valid moves",
#             [i for i in range(tictactoe.action_size) if valid_moves[i] == 1],
#         )
#         action = int(input(f"{player}:"))

#         if valid_moves[action] == 0:
#             print("action not valid")
#             continue
#     else:
#         neutral_state = tictactoe.change_perspective(state, player)
#         mcts_probs = mcts.search(neutral_state)
#         action = np.argmax(mcts_probs)

#     state = tictactoe.get_next_state(state, action, player)

#     value, is_terminal = tictactoe.get_value_and_terminated(state, action)

#     if is_terminal:
#         print(state)
#         if value == 1:
#             print(player, "won")
#         else:
#             print("draw")
#         break

#     player = tictactoe.get_opponent(player)

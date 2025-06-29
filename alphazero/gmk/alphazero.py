import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from gomoku import GameState, Gomoku
from pvmcts import PVMCTS


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game: Gomoku = game
        self.args = args
        self.mcts = PVMCTS(game, args, model)

    def selfPlay(self):
        memory = []
        state: GameState = self.game.get_initial_state()
        player = state.next_player
        while True:
            action_probs = self.mcts.search(state)
            memory.append((state, action_probs, player))

            flat_idx = np.random.choice(self.game.action_size, p=action_probs)

            # 평탄 인덱스 → 2D 좌표
            x = flat_idx % self.game.col_count  # 열
            y = flat_idx // self.game.col_count  # 행
            action = (x, y)

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value
                    returnMemory.append((hist_state, hist_action_probs, hist_outcome))
                return returnMemory
            player = state.next_player

    def train(self, memory):
        random.shuffle(memory)
        bsz = self.args["batch_size"]

        for start in range(0, len(memory), bsz):
            sample = memory[start : start + bsz]

            raw_states, policy_targets, value_targets = zip(*sample)

            enc_states = [self.game.get_encoded_state(s) for s in raw_states]
            state = torch.tensor(np.array(enc_states), dtype=torch.float32)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32)
            value_targets = torch.tensor(
                np.array(value_targets).reshape(-1, 1), dtype=torch.float32
            )

            out_policy, out_value = self.model(state)

            # ① 확률 분포 그대로 사용
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            print(policy_loss.item(), value_loss.item())
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            memory = []

            for selfPlay_iteration in trange(self.args["num_selfPlay_iterations"]):
                memory += self.selfPlay()

            self.model.train()
            for epoch in range(self.args["num_epochs"]):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")

import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from gomoku import GameState, Gomoku
from policy_value_net import PolicyValueNet
from pvmcts_parallel import PVMCTSParallel


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model: PolicyValueNet = model
        self.optimizer = optimizer
        self.game: Gomoku = game
        self.args = args
        self.mcts = PVMCTSParallel(game, args, model)

    def selfPlay(self):
        self.model.eval()
        with torch.no_grad():
            return_memory = []
            # state: GameState = self.game.get_initial_state()
            # player = state.next_player
            # turn = 0
            spGames = [
                SelfPlayGame(self.game)
                for spg in range(self.args["num_parallel_games"])
            ]

            while len(spGames) > 0:
                states = np.stack([spg.state for spg in spGames])

                self.mcts.search(states, spGames)

                for i in range(len(spGames))[::-1]:
                    spg = spGames[i]

                    player = spg.state.next_player  # 현재 턴의 플레이어

                    action_probs = np.zeros(self.game.action_size)
                    for child in spg.root.children:
                        x, y = child.action_taken
                        idx = x + y * self.game.col_count
                        action_probs[idx] = child.visit_count
                    action_probs /= np.sum(action_probs)
                    # print(action_probs)

                    spg.memory.append((spg.root.state, action_probs, player))
                    if spg.turn < self.args["exploration_turns"]:
                        temperature_action_probs = np.maximum(
                            action_probs, 1e-8
                        )  # ε 보정
                        temperature_action_probs = temperature_action_probs ** (
                            1 / self.args["temperature"]
                        )
                        temperature_action_probs /= (
                            temperature_action_probs.sum()
                        )  # ← 정규화 추가
                        flat_idx = np.random.choice(
                            self.game.action_size, p=temperature_action_probs
                        )
                    else:
                        flat_idx = np.argmax(action_probs)

                    # 평탄 인덱스 → 2D 좌표
                    x = flat_idx % self.game.col_count  # 열
                    y = flat_idx // self.game.col_count  # 행
                    action = (x, y)

                    spg.state = self.game.get_next_state(spg.state, action, player)
                    spg.turn += 1

                    value, is_terminal = self.game.get_value_and_terminated(
                        spg.state, action
                    )

                    if is_terminal:
                        for hist_state, hist_action_probs, hist_player in spg.memory:
                            hist_outcome = value if hist_player == player else -value
                            return_memory.append(
                                (hist_state, hist_action_probs, hist_outcome)
                            )

                        del spGames[i]

            return return_memory

    def train(self, memory):
        random.shuffle(memory)
        bsz = self.args["batch_size"]

        self.model.train()
        for start in range(0, len(memory), bsz):
            sample = memory[start : start + bsz]

            raw_states, policy_targets, value_targets = zip(*sample)

            enc_states = self.game.get_encoded_state(list(raw_states))
            state = torch.tensor(
                np.array(enc_states), dtype=torch.float32, device=self.model.device
            )
            policy_targets = torch.tensor(
                np.array(policy_targets), dtype=torch.float32, device=self.model.device
            )
            value_targets = torch.tensor(
                np.array(value_targets).reshape(-1, 1),
                dtype=torch.float32,
                device=self.model.device,
            )

            out_policy, out_value = self.model(state)

            # ① 확률 분포 그대로 사용
            policy_loss = F.cross_entropy(out_policy, policy_targets, reduction="mean")
            value_loss = F.mse_loss(out_value, value_targets)
            # print(policy_loss.item(), value_loss.item())
            loss = policy_loss + value_loss

            # print(
            #     f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Total Loss: {loss.item():.4f}"
            # )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            memory = []

            for selfPlay_iteration in trange(
                self.args["num_selfPlay_iterations"] // self.args["num_parallel_games"]
            ):
                memory += self.selfPlay()

            self.model.train()
            for epoch in range(self.args["num_epochs"]):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")


class SelfPlayGame:
    def __init__(self, game: Gomoku):
        self.state: GameState = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None
        self.turn = 0

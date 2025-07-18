import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from arena import Arena
from gomoku import GameState, Gomoku
from policy_value_net import PolicyValueNet
from pvmcts import PVMCTS


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model: PolicyValueNet = model
        self.optimizer = optimizer
        self.game: Gomoku = game
        self.args = args
        self.mcts = PVMCTS(game, args, model)

    def selfPlay(self):
        self.model.eval()
        with torch.no_grad():
            memory = []
            state: GameState = self.game.get_initial_state()
            player = state.next_player
            turn = 0

            while True:
                action_probs = self.mcts.search(state)
                memory.append((state, action_probs, player))
                if turn < self.args["exploration_turns"]:
                    temperature_action_probs = np.maximum(action_probs, 1e-8)  # ε 보정
                    temperature_action_probs = action_probs ** (
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

                state = self.game.get_next_state(state, action, player)
                turn += 1

                value, is_terminal = self.game.get_value_and_terminated(state, action)

                if is_terminal:
                    returnMemory = []
                    for hist_state, hist_action_probs, hist_player in memory:
                        hist_outcome = value if hist_player == player else -value
                        returnMemory.append(
                            (hist_state, hist_action_probs, hist_outcome)
                        )
                    return returnMemory
                player = state.next_player

    def train(self, memory):
        random.shuffle(memory)
        bsz = self.args["batch_size"]

        self.model.train()
        for start in range(0, len(memory), bsz):
            sample = memory[start : start + bsz]

            raw_states, policy_targets, value_targets = zip(*sample)

            enc_states = [self.game.get_encoded_state(s) for s in raw_states]
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
            loss = policy_loss + value_loss

            # print(
            #     f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Total Loss: {loss.item():.4f}"
            # )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # def learn(self):
    #     for iteration in range(self.args["num_iterations"]):
    #         memory = []

    #         for selfPlay_iteration in trange(self.args["num_selfPlay_iterations"]):
    #             memory += self.selfPlay()

    #         self.model.train()
    #         for epoch in range(self.args["num_epochs"]):
    #             self.train(memory)

    #         torch.save(self.model.state_dict(), f"model_{iteration}.pt")
    #         torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")

    def learn(self):
        # 챔피언 모델과 도전자 모델의 파일 경로 정의
        champion_model_path = "champion.pt"
        challenger_model_path = "challenger.pt"

        # 만약 챔피언 모델이 이미 존재하면, 불러옵니다.
        if os.path.exists(champion_model_path):
            print(f"Loading existing champion model from {champion_model_path}")
            self.model.load_state_dict(torch.load(champion_model_path))
        else:
            print(
                "No champion model found. Starting from scratch and saving initial model."
            )
            torch.save(self.model.state_dict(), champion_model_path)

        # Arena 인스턴스 생성
        arena = Arena(self.game, self.args)

        for i in range(self.args["num_iterations"]):
            print(f"--- Iteration {i + 1} / {self.args['num_iterations']} ---")

            # 1. Self-Play: 현재 챔피언 모델로 데이터를 생성합니다.
            # (매번 최신 챔피언 모델을 다시 불러와서 데이터 생성 시작)
            self.model.load_state_dict(torch.load(champion_model_path))
            self.model.eval()

            memory = []
            for _ in trange(self.args["num_selfPlay_iterations"], desc="Self-Playing"):
                memory += self.selfPlay()

            # 2. Train: 생성된 데이터로 새 모델(도전자)을 훈련합니다.
            self.model.train()
            for _ in trange(self.args["num_epochs"], desc="Training"):
                self.train(memory)

            # 훈련된 도전자 모델을 임시 저장
            torch.save(self.model.state_dict(), challenger_model_path)

            # 3. Evaluate: 새로운 도전자와 기존 챔피언을 비교 평가합니다.
            print("\n--- Evaluating New Model (Challenger) vs. Champion ---")

            # 도전자 모델과 챔피언 모델을 Arena에서 사용할 수 있도록 준비
            challenger = self.model  # 현재 self.model이 바로 도전자
            champion = PolicyValueNet(
                self.game,
                self.args["num_planes"],
                self.args["num_resblocks"],
                self.args["num_hidden"],
                self.model.device,
            )
            champion.load_state_dict(torch.load(champion_model_path))

            # Arena를 통해 대결 진행
            win_rate = arena.evaluate(challenger, champion)
            print(
                f"\nChallenger Win Rate: {win_rate:.2f} (Required: > {self.args['eval_win_rate']})"
            )

            # 4. Select: 승률에 따라 챔피언을 교체할지 결정합니다.
            if win_rate > self.args["eval_win_rate"]:
                print("🏆 New model is stronger! Promoting to Champion.")
                # 도전자 모델이 새로운 챔피언이 됨
                torch.save(challenger.state_dict(), champion_model_path)
            else:
                print(" Challenger is not strong enough. Keeping the old Champion.")
                # 변경사항 없음, 다음 iteration에서 기존 챔피언으로 다시 self-play 진행

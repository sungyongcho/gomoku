from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from ai.policy_value_net import PolicyValueNet
from ai.pv_mcts import PVMCTS  # 업데이트된 PVMCTS 포함
from core.game_config import DRAW, LOSE, PLAYER_1, PLAYER_2, WIN
from core.gomoku import Gomoku


@dataclass
class SelfPlayConfig:
    sims: int = 200
    c_puct: float = 1.4
    temperature_turns: int = 20
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    resign_threshold: float = -0.8  # value 예측 Q
    max_turns: int = 400  # 안전 캡
    device: str = "cpu"


# 샘플 타입
State = np.ndarray  # (C, 19, 19)
Pi = np.ndarray  # (19, 19)
Z = float
Sample = Tuple[State, Pi, Z]


def play_one_game(model: torch.nn.Module, cfg: SelfPlayConfig) -> List[Sample]:
    """MCTS 로 한 판 자가대국 → (state, π, z) 리스트 반환"""

    mcts = PVMCTS(model, sims=cfg.sims, c_puct=cfg.c_puct, device=cfg.device)
    game = Gomoku()
    history: List[Tuple[State, Pi]] = []

    turn = 0
    while game.winner is None and turn < cfg.max_turns:
        # 1) MCTS 탐색
        root = mcts.search(game.board)

        # 2) Dirichlet noise (root prior)
        PVMCTS.apply_dirichlet_noise(root, cfg.dirichlet_alpha, cfg.dirichlet_epsilon)

        # 3) π 및 수 선택 (temperature)
        move, pi = mcts.get_move_and_pi(root)
        if turn < cfg.temperature_turns:
            move = PVMCTS.sample_with_temperature(pi, cfg.temperature)

        # 4) 상태 기록 (move 이전의 state)
        state_planes = game.board.to_tensor().squeeze(0).numpy()
        history.append((state_planes, pi))

        # 5) 실제 착수
        game.play_move(*move)
        turn += 1

        # 6) 조기 기권 조건
        if turn > 30 and root.Q < cfg.resign_threshold:
            game.winner = PLAYER_2 if game.current_player == PLAYER_1 else PLAYER_1

    # 최종 승패 값(z) 계산
    if game.winner == PLAYER_1:
        z_final = WIN
    elif game.winner == PLAYER_2:
        z_final = LOSE
    else:
        z_final = DRAW

    return [(s, p, z_final) for s, p in history]


class SelfPlayWorker(mp.Process):
    """독립 프로세스에서 자가대국을 생성해 Queue 에 샘플 리스트 push"""

    def __init__(
        self, cfg: SelfPlayConfig, queue: mp.Queue, shared_model: torch.nn.Module
    ):
        super().__init__()
        self.cfg = cfg
        self.queue = queue
        # 로컬 모델: shared_model state_dict 복사
        self.model = PolicyValueNet().to(cfg.device)
        self.model.load_state_dict(shared_model.state_dict())
        torch.set_num_threads(1)

    def run(self):
        while True:
            samples = play_one_game(self.model, self.cfg)
            self.queue.put(samples)  # 한 게임 분량 전송

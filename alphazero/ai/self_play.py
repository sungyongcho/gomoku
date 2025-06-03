# ai/self_play.py
# ------------------------------------------------------------
# 1판을 MCTS + 현 네트워크로 두고
# (state_planes, π, z) 경험을 ReplayBuffer 에 저장
# ------------------------------------------------------------
from typing import List

import numpy as np
from ai.pv_mcts import PVMCTS
from ai.replay_buffer import ReplayBuffer
from ai.state_encoder import encode
from core.game_config import DRAW, LOSE, WIN
from core.gomoku import Gomoku
from core.rules.terminate import is_terminal


def play_one_game(
    mcts: PVMCTS,
    buffer: ReplayBuffer,
    temp_turns: int = 15,
) -> int:
    """
    1판 셀프플레이를 수행하고 버퍼에 경험을 push

    Returns
    -------
    winner :  1  (흑) / 2 (백) / 0 (무승부)
    """
    game = Gomoku()
    states: List[np.ndarray] = []
    policies: List[np.ndarray] = []

    while True:
        # ─ 1) MCTS 탐색
        root = mcts.search(game.board)
        best_move, pi = mcts.get_move_and_pi(root)  # pi (N,N)

        # ─ 2) 데이터 저장용
        states.append(encode(game.board))  # (C,N,N)
        policies.append(pi)

        # ─ 3) temperature 스케줄
        move = (
            np.unravel_index(np.random.choice(pi.size, p=pi.flatten()), pi.shape)
            if game.turn < temp_turns
            else best_move
        )

        # ─ 4) 수 두기
        game.board.set_value(
            move[1], move[0], game.board.next_player
        )  # Gomoku 내부에서 규칙 처리

        # ─ 5) 종료 체크
        check_terminal = is_terminal(game.board)
        if check_terminal is not None or check_terminal == DRAW:
            winner = 0 if check_terminal == DRAW else game.winner
            z_final = {
                game.board.last_player: WIN,
                game.board.next_player: LOSE,
                0: DRAW,
            }[winner]
            break

    # ─ 6) ReplayBuffer에 push
    for idx, s in enumerate(states):
        # 플레이어 시점 맞추어 ±
        z = z_final if idx % 2 == 0 else -z_final
        buffer.add(s, policies[idx], z)

    return winner

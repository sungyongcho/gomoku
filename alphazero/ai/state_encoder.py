import numpy as np
import torch
from core.board import Board
from core.game_config import CAPTURE_GOAL, PLAYER_2


def encode(board: Board) -> torch.Tensor:
    # TODO: Check if batch needed
    """
    0 | last player stones  | [0, 1]
    1 | next player stones  | [0, 1]
    2 | last played stones of the game | 0 / 1
    3 | turn indicator -- whether board.next_player will be player(t/f) | 0 / 1
    4 | capture score ratio of last player by goal | 0 ~ 1
    5 | capture score ratio of next player by goal | 0 ~ 1
    """
    pos = board.pos  # (N, N) uint8

    # 돌 분리
    last_board = (pos == board.last_player).astype(np.float32)
    next_board = (pos == board.next_player).astype(np.float32)

    # 마지막 수 평면
    last_stone = np.zeros_like(pos, dtype=np.float32)
    if getattr(board, "last_x", None) is not None:
        last_stone[board.last_y, board.last_x] = 1.0

    # 턴 표시 (다음 차례가 last_player이면 1, 아니면 0)
    turn_plane = np.full_like(
        pos, 1.0 if board.next_player == PLAYER_2 else 0.0, dtype=np.float32
    )

    cap_last = np.full_like(pos, board.last_pts / CAPTURE_GOAL, dtype=np.float32)
    cap_next = np.full_like(pos, board.next_pts / CAPTURE_GOAL, dtype=np.float32)

    # print(turn_plane)

    stacked = np.stack(
        [
            last_board,
            next_board,
            last_stone,
            turn_plane,
            cap_last,
            cap_next,
        ]
    )  # (6, N, N)

    return torch.from_numpy(stacked)  # float32

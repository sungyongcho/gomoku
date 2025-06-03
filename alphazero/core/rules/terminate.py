import numpy as np
from core.board import Board
from core.game_config import EMPTY_SPACE, NUM_LINES
from core.rules.doublethree import detect_doublethree


def check_local_gomoku(board: Board, x: int, y: int, player: int) -> bool:
    """Check if (x, y) forms 5-in-a-row using NumPy slicing."""
    array = board.pos
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for dx, dy in directions:
        coords_x = np.clip(np.arange(-4, 5) * dx + x, 0, NUM_LINES - 1)
        coords_y = np.clip(np.arange(-4, 5) * dy + y, 0, NUM_LINES - 1)

        line = array[coords_y, coords_x]  # Extract the line
        if len(line) >= 5 and np.any(
            np.convolve(line == player, np.ones(5, dtype=int), mode="valid") == 5
        ):
            return True

    return False


def board_is_functionally_full(board: Board) -> bool:
    """Check if the board is full or all moves are forbidden."""
    empty_positions = board.pos == EMPTY_SPACE
    if np.any(empty_positions):
        for player in [board.last_player, board.next_player]:
            if np.any(
                [
                    not detect_doublethree(board, col, row, player)
                    for col, row in zip(*np.where(empty_positions))
                ]
            ):
                return False
    return True


def is_won_by_score(board: Board, player: int) -> bool:
    """Check if the given player has won by reaching the score goal."""
    target_score = board.last_pts if player == board.last_player else board.next_pts
    return target_score >= board.goal


def is_terminal(board: Board) -> int | None:
    """
    승자를 리턴 (PLAYER_1, PLAYER_2) / 무승부·미종료면 None
    - 5목 완성 or 캡처 점수로 승리
    - 동시승 상황은 게임 규칙에 맞게 조정 (여기선 '마지막 수' 우선 승리로 가정)
    """
    if board.last_x is None or board.last_y is None:
        return None
    # 1) 마지막 수 둔 플레이어가 5목 or 캡처 승
    if check_local_gomoku(
        board, board.last_x, board.last_y, board.last_player
    ) or is_won_by_score(board, board.last_player):
        return board.last_player

    # 2) 상대 플레이어가 이미 5목 완성 상태였는지 (이론적 안전장치)
    opp_player = board.next_player
    if check_local_gomoku(
        board, board.last_x, board.last_y, opp_player
    ) or is_won_by_score(board, opp_player):
        return opp_player

    # 3) 무승부 (둘 곳 없음)
    if board_is_functionally_full(board):
        return 0  # 0을 무승부 코드로 사용

    # 4) 아직 승자 없음
    return None

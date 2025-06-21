from typing import Dict, List

import numpy as np

from game_config import DIRECTIONS, NUM_LINES, PLAYER_1, PLAYER_2, get_pos

CAPTURE_WINDOW: int = 3


def dfs_capture(
    board_pos: np.ndarray,
    x: int,
    y: int,
    player: int,
    direction: tuple[int, int],
    count: int,
) -> bool:
    nx = x + direction[0]
    ny = y + direction[1]

    if count == CAPTURE_WINDOW:
        return get_pos(board_pos, nx, ny) == player

    opponent = PLAYER_2 if player == PLAYER_1 else PLAYER_1
    if get_pos(board_pos, nx, ny) != opponent:
        return False

    return dfs_capture(board_pos, nx, ny, player, direction, count + 1)


def detect_captured_stones(
    board_pos: np.ndarray, x: int, y: int, player: int
) -> List[Dict[str, int]]:
    captured_stones = []

    for dir in DIRECTIONS:
        nx, ny = x + dir[0] * CAPTURE_WINDOW, y + dir[1] * CAPTURE_WINDOW
        if not (0 <= nx < NUM_LINES and 0 <= ny < NUM_LINES):
            continue

        if dfs_capture(board_pos, x, y, player, dir, 1):
            captured_stones.append(
                {
                    "x": x + dir[0],
                    "y": y + dir[1],
                    "stone": get_pos(board_pos, x + dir[0], y + dir[1]),
                }
            )
            captured_stones.append(
                {
                    "x": x + (dir[0] * 2),
                    "y": y + (dir[1] * 2),
                    "stone": get_pos(board_pos, x + (dir[0] * 2), y + (dir[1] * 2)),
                }
            )

    return captured_stones

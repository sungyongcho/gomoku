from constants import DIRECTIONS, NUM_LINES, PLAYER_1, PLAYER_2
from services.board import Board

PLAYER_1 = 1
PLAYER_2 = 2
EMPTY_SPACE = 0


def dfs_capture(
    board: Board, x: int, y: int, player: int, direction: tuple[int, int], count: int
) -> bool:
    nx = x + direction[0]
    ny = y + direction[1]

    if count == 3:
        return board.get_value(nx, ny) == player

    opponent = PLAYER_2 if player == PLAYER_1 else PLAYER_1
    if board.get_value(nx, ny) != opponent:
        return False

    return dfs_capture(board, nx, ny, player, direction, count + 1)


def capture_opponent(board: Board, x: int, y: int, player: int):
    captured_stones = []

    for dir in DIRECTIONS:
        nx, ny = x + dir[0] * 3, y + dir[1] * 3
        if not (0 <= nx < NUM_LINES and 0 <= ny < NUM_LINES):
            continue

        if dfs_capture(board, x, y, player, dir, 1):
            captured_stones.append(
                {
                    "x": x + dir[0],
                    "y": y + dir[1],
                    "stone": board.get_value(x + dir[0], y + dir[1]),
                }
            )
            captured_stones.append(
                {
                    "x": x + (dir[0] * 2),
                    "y": y + (dir[1] * 2),
                    "stone": board.get_value(x + (dir[0] * 2), y + (dir[1] * 2)),
                }
            )

    return captured_stones

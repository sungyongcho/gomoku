from typing import Tuple

from config import *
from services.board import Board


def is_within_bounds(x: int, y: int, offset_x: int, offset_y: int) -> bool:
    """Check if the given coordinates with offsets are within board bounds."""
    return 0 <= x + offset_x < NUM_LINES and 0 <= y + offset_y < NUM_LINES


def check_middle(
    board: Board,
    x: int,
    y: int,
    dir: Tuple[int, int],
    player: str,
) -> bool:
    """Check for an open three pattern with a middle gap."""
    # Ensure all required cells are within bounds
    if not is_within_bounds(x, y, dir[0] * 3, dir[1] * 3):
        return False

    # Check the pattern in the opposite direction and the current sequence
    if not (
        board.get_value(x - dir[0], y - dir[1]) == player
        and board.get_value(x - dir[0] * 2, y - dir[1] * 2) == EMPTY_SPACE
    ):
        return False

    # Construct the line and check for specific patterns
    line = "".join(
        [
            board.get_value(x + dir[0], y + dir[1]),
            board.get_value(x + dir[0] * 2, y + dir[1] * 2),
            board.get_value(x + dir[0] * 3, y + dir[1] * 3),
        ]
    )
    return line[:2] == f"{player}." or line == f".{player}."


def dfs(
    board: Board,
    x: int,
    y: int,
    dir: Tuple[int, int],
    player: str,
    count: int,
    player_count=1,
) -> bool:
    """Perform a depth-first search to detect open three patterns."""
    nx, ny = x + dir[0], y + dir[1]
    opponent = PLAYER_1 if player == PLAYER_2 else PLAYER_2

    # Ensure (nx, ny) is within bounds
    if not is_within_bounds(x, y, dir[0], dir[1]):
        return False

    # Handle cases for count == 3 and count == 4
    if count == 3:
        return player_count == 3 and board.get_value(nx, ny) == EMPTY_SPACE

    if count == 4:
        return player_count == 3 and board.get_value(nx, ny) == EMPTY_SPACE

    # Stop if the next cell is occupied by the opponent
    if board.get_value(nx, ny) == opponent:
        return False

    # Recursive cases
    if board.get_value(nx, ny) == player:
        return dfs(board, nx, ny, dir, player, count + 1, player_count + 1)

    return dfs(board, nx, ny, dir, player, count + 1)


def check_doublethree(board: Board, x: int, y: int, player: str) -> bool:
    """Check if placing a stone creates a double three."""
    print("doublethree checking")
    openthree_count = 0

    for dir in DIRECTIONS:
        # Skip if the current direction or opposite direction is out of bounds
        if not is_within_bounds(x, y, dir[0], dir[1]) or not is_within_bounds(
            x, y, -dir[0], -dir[1]
        ):
            continue

        # Skip if the opposite cell is blocked by an opponent's stone
        opponent = PLAYER_1 if player == PLAYER_2 else PLAYER_2
        if board.get_value(x - dir[0], y - dir[1]) == opponent:
            continue

        # Check for open three patterns
        if check_middle(board, x, y, dir, player):
            print(f"Open three detected using middle check in direction {dir}.")
            openthree_count += 1
            continue

        if board.get_value(x - dir[0], y - dir[1]) == EMPTY_SPACE and dfs(
            board, x, y, dir, player, 1
        ):
            print(f"Open three detected using DFS in direction {dir}.")
            openthree_count += 1

    return openthree_count >= 2

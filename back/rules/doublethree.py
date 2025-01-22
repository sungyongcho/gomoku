from typing import Tuple

from constants import DIRECTIONS, NUM_LINES, PLAYER_1, PLAYER_2, UNIQUE_DIRECTIONS
from services.board import Board


def is_within_bounds(x: int, y: int, offset_x: int, offset_y: int) -> bool:
    """Check if the given coordinates with offsets are within board bounds."""
    return 0 <= x + offset_x < NUM_LINES and 0 <= y + offset_y < NUM_LINES


def get_line(board: Board, x: int, y: int, dir: Tuple[int, int], length: int) -> str:
    """Construct a line of cells in the given direction up to the specified length."""

    result = []
    for i in range(1, length + 1):
        if not is_within_bounds(x, y, (dir[0] * i), (dir[1] * i)):
            return ""
        new_x = x + dir[0] * i
        new_y = y + dir[1] * i
        result.append(board.get_value(new_x, new_y))
    return "".join(result)


def check_middle(
    board: Board, x: int, y: int, dir: Tuple[int, int], player: str, opponent: str
) -> bool:
    """Check for an open three pattern with or without a middle gap."""
    line = get_line(board, x, y, dir, 3)
    line_opposite = get_line(board, x, y, (-dir[0], -dir[1]), 3)

    if len(line) < 3 or len(line_opposite) < 3:
        return False

    # Case 1: .O$O., but not X.O$O.X and O$O.O
    if not (
        (line_opposite == f"{player}.{opponent}" and line == f"{player}.{opponent}")
        or (line_opposite[0] == f"{player}" and line == f"{player}.{player}")
    ) and (line_opposite[0:2] == f"{player}." and line[0:2] == f"{player}."):
        return True

    if line_opposite[0:2] == f"{player}." and line == f".{player}.":
        return True

    return False


# def dfs(
#     board: Board,
#     x: int,
#     y: int,
#     dir: Tuple[int, int],
#     player: str,
#     count: int,
#     player_count=1,
# ) -> bool:
#     """Perform a depth-first search to detect open three patterns."""
#     nx, ny = x + dir[0], y + dir[1]
#     opponent = PLAYER_1 if player == PLAYER_2 else PLAYER_2

#     # Ensure (nx, ny) is within bounds
#     if not is_within_bounds(x, y, dir[0], dir[1]):
#         return False

#     # Handle cases for count == 3 and count == 4
#     if count == 3:
#         return player_count == 3 and board.get_value(nx, ny) == EMPTY_SPACE

#     if count == 4:
#         return player_count == 3 and board.get_value(nx, ny) == EMPTY_SPACE

#     # Stop if the next cell is occupied by the opponent
#     if board.get_value(nx, ny) == opponent:
#         return False

#     # Recursive cases
#     if board.get_value(nx, ny) == player:
#         return dfs(board, nx, ny, dir, player, count + 1, player_count + 1)

#     return dfs(board, nx, ny, dir, player, count + 1)


def check_edge(
    board: Board, x: int, y: int, dir: Tuple[int, int], player: str, opponent: str
) -> bool:
    line = get_line(board, x, y, dir, 4)
    line_opposite = get_line(board, x, y, (-dir[0], -dir[1]), 2)

    if len(line) < 4 or len(line_opposite) < 2:
        return False

    # Case 1: .$OO., but not X.$OO.X, $OO.O
    if not (
        (line == f"{player}{player}.{opponent}" and line_opposite == f".{opponent}")
        or line == f"{player}{player}.{player}"
    ) and (line[0:3] == f"{player}{player}." and line_opposite[0] == "."):
        return True

    # Case 2: .$O.O., but not O.$O.O
    if not (line[0:3] == f"{player}.{player}" and line_opposite == f".{player}") and (
        line == f"{player}.{player}." and (line_opposite[0] == ".")
    ):
        return True

    # Case 3: .$.OO.
    if line == f".{player}{player}." and line_opposite[0] == ".":
        return True

    return False


def check_doublethree(board: Board, x: int, y: int, player: str) -> bool:
    """Check if placing a stone creates a double three."""
    # print("doublethree checking")
    openthree = []

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

        # Check for open three patterns at the edge
        if check_edge(board, x, y, dir, player, opponent):
            # print(f"Open three detected using edge check in direction {dir}.")
            openthree.append(("edge", dir))
            continue

        # Check for open three patterns in the middle (limit to unique directions)
        if dir in UNIQUE_DIRECTIONS:
            if check_middle(board, x, y, dir, player, opponent):
                # print(f"Open three detected using middle check in direction {dir}.")
                openthree.append(("middle", dir))

    # Debugging output (you can remove this later)
    # print(f"Open three count: {len(openthree)} (Details: {openthree})")

    # Return whether a double three is detected
    return len(openthree) >= 2

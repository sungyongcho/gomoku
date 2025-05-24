from typing import Tuple

import numpy as np
from constants import (
    DIRECTIONS,
    NUM_LINES,
    UNIQUE_DIRECTIONS,
)
from services.board import Board

PLAYER_1 = 1
PLAYER_2 = 2
EMPTY_SPACE = 0


def is_within_bounds(x: int, y: int, offset_x: int, offset_y: int) -> bool:
    """Check if the given coordinates with offsets are within board bounds."""
    return 0 <= x + offset_x < NUM_LINES and 0 <= y + offset_y < NUM_LINES


def get_line(
    board: Board, x: int, y: int, dir: Tuple[int, int], length: int
) -> np.ndarray:
    """Construct a line of cells in the given direction up to the specified length."""
    result = []
    for i in range(1, length + 1):
        if not is_within_bounds(x, y, dir[0] * i, dir[1] * i):
            return np.array([], dtype=np.uint8)
        new_x = x + dir[0] * i
        new_y = y + dir[1] * i
        result.append(board.get_value(new_x, new_y))
    return np.array(result, dtype=np.uint8)


def check_middle(
    board: Board, x: int, y: int, dir: Tuple[int, int], player: int, opponent: int
) -> bool:
    """Check for an open three pattern with or without a middle gap."""
    line = get_line(board, x, y, dir, 3)
    line_opposite = get_line(board, x, y, (-dir[0], -dir[1]), 3)

    if len(line) < 3 or len(line_opposite) < 3:
        return False

    # Case 1: Check patterns numerically
    if not (
        (
            line_opposite.tolist() == [player, EMPTY_SPACE, opponent]
            and line.tolist() == [player, EMPTY_SPACE, opponent]
        )
        or (
            line_opposite[0] == player
            and line.tolist() == [player, EMPTY_SPACE, player]
        )
    ) and (
        line_opposite[:2].tolist() == [player, EMPTY_SPACE]
        and line[:2].tolist() == [player, EMPTY_SPACE]
    ):
        return True

    if line_opposite[:2].tolist() == [player, EMPTY_SPACE] and line.tolist() == [
        EMPTY_SPACE,
        player,
        EMPTY_SPACE,
    ]:
        return True

    return False


def check_edge(
    board: Board, x: int, y: int, dir: Tuple[int, int], player: int, opponent: int
) -> bool:
    line = get_line(board, x, y, dir, 4)
    line_opposite = get_line(board, x, y, (-dir[0], -dir[1]), 2)

    if len(line) < 4 or len(line_opposite) < 2:
        return False

    # Case 1: .$OO., but not X.$OO.X, $OO.O
    if not (
        (
            line.tolist() == [player, player, EMPTY_SPACE, opponent]
            and line_opposite.tolist() == [EMPTY_SPACE, opponent]
        )
        or line.tolist() == [player, player, EMPTY_SPACE, player]
    ) and (
        line[:3].tolist() == [player, player, EMPTY_SPACE]
        and line_opposite[0] == EMPTY_SPACE
    ):
        return True

    # Case 2: .$O.O., but not O.$O.O
    if not (
        line[:3].tolist() == [player, EMPTY_SPACE, player]
        and line_opposite.tolist() == [EMPTY_SPACE, player]
    ) and (
        line.tolist() == [player, EMPTY_SPACE, player, EMPTY_SPACE]
        and line_opposite[0] == EMPTY_SPACE
    ):
        return True

    # Case 3: .$.OO.
    if (
        line.tolist() == [EMPTY_SPACE, player, player, EMPTY_SPACE]
        and line_opposite[0] == EMPTY_SPACE
    ):
        return True

    return False


def check_doublethree(board: Board, x: int, y: int, player: int) -> bool:
    """Check if placing a stone creates a double three."""
    openthree = []
    opponent = PLAYER_1 if player == PLAYER_2 else PLAYER_2

    for dir in DIRECTIONS:
        if not is_within_bounds(x, y, dir[0], dir[1]) or not is_within_bounds(
            x, y, -dir[0], -dir[1]
        ):
            continue

        if board.get_value(x - dir[0], y - dir[1]) == opponent:
            continue

        if check_edge(board, x, y, dir, player, opponent):
            openthree.append(("edge", dir))
            continue

        if dir in UNIQUE_DIRECTIONS:
            if check_middle(board, x, y, dir, player, opponent):
                openthree.append(("middle", dir))

    return len(openthree) >= 2

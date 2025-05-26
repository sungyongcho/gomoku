from typing import Tuple

import numpy as np
from core.board import Board
from core.game_config import (
    DIRECTIONS,
    EMPTY_SPACE,
    NUM_LINES,
    UNIQUE_DIRECTIONS,
    opponent_player,
)


def is_within_bounds(x: int, y: int, offset: Tuple) -> bool:
    """Check if the given coordinates with offsets are within board bounds."""
    return 0 <= x + offset[0] < NUM_LINES and 0 <= y + offset[1] < NUM_LINES


def get_line(
    board: Board, x: int, y: int, dir: Tuple[int, int], length: int
) -> np.ndarray:
    """Construct a line of cells in the given direction up to the specified length."""
    result = []
    for i in range(1, length + 1):
        if not is_within_bounds(x, y, dir):
            return np.array([], dtype=np.uint8)
        new_x = x + dir[0] * i
        new_y = y + dir[1] * i
        result.append(board.get_value(new_x, new_y))
    return np.array(result, dtype=np.uint8)


def check_middle_1(
    board: Board, x: int, y: int, dir: Tuple[int, int], player: int, opponent: int
) -> bool:
    line = get_line(board, x, y, dir, 3)
    line_opposite = get_line(board, x, y, (-dir[0], -dir[1]), 3)

    if len(line) < 3 or len(line_opposite) < 3:
        return False

    # Case: .O$O., but not X.O$O.X, O$O.O
    if not (
        (
            line.tolist() == [player, EMPTY_SPACE, opponent]
            and line_opposite.tolist() == [player, EMPTY_SPACE, opponent]
        )
        or (
            line.tolist() == [player, EMPTY_SPACE, player]
            and line_opposite[0] == player
        )
    ) and (
        line[:2].tolist() == [player, EMPTY_SPACE]
        and line_opposite[:2].tolist() == [player, EMPTY_SPACE]
    ):
        return True

    return False


def check_middle_2(
    board: Board, x: int, y: int, dir: Tuple[int, int], player: int, opponent: int
) -> bool:
    line = get_line(board, x, y, dir, 3)
    line_opposite = get_line(board, x, y, (-dir[0], -dir[1]), 3)
    if len(line) < 3 or len(line_opposite) < 3:
        return False

    # Case: .O$.O.
    if line.tolist() == [
        EMPTY_SPACE,
        player,
        EMPTY_SPACE,
    ] and line_opposite[:2].tolist() == [player, EMPTY_SPACE]:
        return True


def check_edge(
    board: Board, x: int, y: int, dir: Tuple[int, int], player: int, opponent: int
) -> bool:
    line = get_line(board, x, y, dir, 4)
    line_opposite = get_line(board, x, y, (-dir[0], -dir[1]), 2)

    if len(line) < 4 or len(line_opposite) < 2:
        return False

    # Case 1: .$OO. /  but not X.$OO.X, $OO.O
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

    # Case 2: .$O.O. / but not O.$O.O
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


def detect_doublethree(board: Board, x: int, y: int, player: int) -> bool:
    """Check if placing a stone creates a double three."""
    count: int = 0
    opponent = opponent_player(player)

    for dir in DIRECTIONS:
        if not is_within_bounds(x, y, dir):
            continue

        if check_edge(board, x, y, dir, player, opponent):
            count += 1
            continue

        if dir in UNIQUE_DIRECTIONS and check_middle_1(
            board, x, y, dir, player, opponent
        ):
            count += 1
            continue

        if check_middle_2(board, x, y, dir, player, opponent):
            count += 1
            continue

    return count >= 2

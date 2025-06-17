from typing import Tuple

import numpy as np
from core.game_config import (
    DIRECTIONS,
    EMPTY_SPACE,
    NUM_LINES,
    UNIQUE_DIRECTIONS,
    get_pos,
    opponent_player,
)


def is_within_bounds(x: int, y: int, offset: Tuple) -> bool:
    """Check if the given coordinates with offsets are within board bounds."""
    return 0 <= x + offset[0] < NUM_LINES and 0 <= y + offset[1] < NUM_LINES


def get_line(
    board_pos: np.ndarray, x: int, y: int, dir: Tuple[int, int], length: int
) -> np.ndarray:
    """Construct a line of cells in the given direction up to the specified length."""
    result = []
    for i in range(1, length + 1):
        new_x = x + dir[0] * i
        new_y = y + dir[1] * i
        if not (0 <= new_x < NUM_LINES and 0 <= new_y < NUM_LINES):
            break
        result.append(get_pos(board_pos, new_x, new_y))
    return np.array(result, dtype=np.uint8)


def check_middle_1(
    board_pos: np.ndarray,
    x: int,
    y: int,
    dir: Tuple[int, int],
    player: int,
    opponent: int,
) -> bool:
    line = get_line(board_pos, x, y, dir, 3)
    line_opposite = get_line(board_pos, x, y, (-dir[0], -dir[1]), 3)

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
    board_pos: np.ndarray,
    x: int,
    y: int,
    dir: Tuple[int, int],
    player: int,
) -> bool:
    line = get_line(board_pos, x, y, dir, 3)
    line_opposite = get_line(board_pos, x, y, (-dir[0], -dir[1]), 3)
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
    board_pos: np.ndarray,
    x: int,
    y: int,
    dir: Tuple[int, int],
    player: int,
    opponent: int,
) -> bool:
    line = get_line(board_pos, x, y, dir, 4)
    line_opposite = get_line(board_pos, x, y, (-dir[0], -dir[1]), 2)

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


def detect_doublethree(board_pos: np.ndarray, x: int, y: int, player: int) -> bool:
    """Check if placing a stone creates a double three."""
    count: int = 0
    opponent = opponent_player(player)

    for dir in DIRECTIONS:
        if not is_within_bounds(x, y, dir):
            continue

        if check_edge(board_pos, x, y, dir, player, opponent):
            count += 1
            continue

        if dir in UNIQUE_DIRECTIONS and check_middle_1(
            board_pos, x, y, dir, player, opponent
        ):
            count += 1
            continue

        if check_middle_2(board_pos, x, y, dir, player):
            count += 1
            continue

    return count >= 2
# import numpy as np
# from core.game_config import (
#     DIRECTIONS,
#     EMPTY_SPACE,
#     NUM_LINES,
#     PLAYER_1,
#     PLAYER_2,
#     UNIQUE_DIRECTIONS,
# )
# from numba import boolean, int32, njit


# # ───────── 헬퍼 ─────────
# @njit(inline="always")
# def opp_player(p: int32) -> int32:
#     return PLAYER_2 if p == PLAYER_1 else PLAYER_1


# @njit(inline="always")
# def inside(xx: int32, yy: int32) -> boolean:
#     return 0 <= xx < NUM_LINES and 0 <= yy < NUM_LINES


# @njit(inline="always")
# def getp(b: np.ndarray, xx: int32, yy: int32) -> int32:
#     return int32(b[yy, xx])


# @njit(inline="always")
# def edge_pattern(
#     b: np.ndarray, x: int32, y: int32, dx: int32, dy: int32, player: int32
# ) -> boolean:
#     if not (inside(x + 3 * dx, y + 3 * dy) and inside(x - dx, y - dy)):
#         return False

#     a = getp(b, x + dx, y + dy)
#     b1 = getp(b, x + 2 * dx, y + 2 * dy)
#     c = getp(b, x + 3 * dx, y + 3 * dy)
#     bm = getp(b, x - dx, y - dy)

#     # .$O.O.  / .$$O.  / .$.OO.
#     if a == player and b1 == player and c == EMPTY_SPACE and bm == EMPTY_SPACE:
#         return True
#     if a == EMPTY_SPACE and b1 == player and c == EMPTY_SPACE and bm == EMPTY_SPACE:
#         return True
#     if a == EMPTY_SPACE and b1 == player and c == player and bm == EMPTY_SPACE:
#         return True
#     return False


# @njit(inline="always")
# def mid1_pattern(b, x, y, dx, dy, player, opponent):
#     if not (inside(x + 2 * dx, y + 2 * dy) and inside(x - 2 * dx, y - 2 * dy)):
#         return False
#     a1 = getp(b, x + dx, y + dy)
#     a2 = getp(b, x + 2 * dx, y + 2 * dy)
#     b1 = getp(b, x - dx, y - dy)
#     b2 = getp(b, x - 2 * dx, y - 2 * dy)

#     # .O$O.   (양끝 EMPTY + player)
#     cond_core = (
#         a1 == EMPTY_SPACE and a2 == player and b1 == EMPTY_SPACE and b2 == player
#     )
#     # .O$O.X  /  X.O$O.  금지조건
#     forbid = (a2 == opponent) or (b2 == opponent)
#     return cond_core and not forbid


# # ───────── middle-2 (.O$.O.) ─────────
# @njit(inline="always")
# def mid2_pattern(b, x, y, dx, dy, player):
#     if not (inside(x + 2 * dx, y + 2 * dy) and inside(x - 2 * dx, y - 2 * dy)):
#         return False
#     a1 = getp(b, x + dx, y + dy)
#     a2 = getp(b, x + 2 * dx, y + 2 * dy)
#     b1 = getp(b, x - dx, y - dy)
#     b2 = getp(b, x - 2 * dx, y - 2 * dy)
#     # .O$.O.  (player-EMPTY-player, 양끝 EMPTY)
#     return a1 == player and a2 == EMPTY_SPACE and b1 == EMPTY_SPACE and b2 == player


# # ───────── detect_doublethree ─────────
# @njit(fastmath=True, cache=True)
# def detect_doublethree(board_pos, x, y, player):
#     opponent = opp_player(player)
#     threes = 0
#     for dx, dy in DIRECTIONS:
#         if edge_pattern(board_pos, x, y, dx, dy, player):
#             threes += 1
#             if threes >= 2:
#                 return True

#         if (dx, dy) in UNIQUE_DIRECTIONS:
#             if mid1_pattern(board_pos, x, y, dx, dy, player, opponent):
#                 threes += 1
#                 if threes >= 2:
#                     return True
#             if mid2_pattern(board_pos, x, y, dx, dy, player):
#                 threes += 1
#                 if threes >= 2:
#                     return True
#     return False

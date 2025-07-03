# board

import numpy as np

EMPTY_DOT = "."

PLAYER_X = "X"  # Black stone
PLAYER_O = "O"  # White stone
EMPTY_SPACE = 0
PLAYER_1 = 1  # Black stone
PLAYER_2 = 2  # White stone
OUT_OF_BOUNDS = 3

NUM_LINES = 5
CAPTURE_GOAL = 2
GOMOKU_GOAL = 4


def opponent_player(player):
    return PLAYER_2 if player == PLAYER_1 else PLAYER_1


def get_pos(board_pos: np.ndarray, x: int, y: int) -> int:
    """Get the value at a specific column and row."""
    return board_pos[y, x]


def set_pos(board_pos: np.ndarray, x: int, y: int, value: int) -> None:
    """Get the value at a specific column and row."""
    board_pos[y, x] = value


def convert_index_to_coordinates(col: int, row: int) -> str:
    if not (0 <= col < NUM_LINES):
        raise ValueError(f"Column index must be between 0 and {NUM_LINES}.")
    if not (0 <= row < NUM_LINES):
        raise ValueError(f"Row index must be between 0 and {NUM_LINES}.")
    col_char = chr(ord("A") + col)
    return f"{col_char}{row + 1}"


# direction for doublethree
NORTH = (0, -1)
NORTHEAST = (1, -1)
EAST = (1, 0)
SOUTHEAST = (1, 1)
SOUTH = (0, 1)
SOUTHWEST = (-1, 1)
WEST = (-1, 0)
NORTHWEST = (-1, -1)

DIRECTIONS = (NORTH, NORTHEAST, EAST, SOUTHEAST, SOUTH, SOUTHWEST, WEST, NORTHWEST)

UNIQUE_DIRECTIONS = (
    NORTH,
    NORTHEAST,
    EAST,
    SOUTHEAST,
)


def calc_num_hidden(num_lines: int, min_ch: int = 32, max_ch: int = 128) -> int:
    """
    보드 한 변(N)을 받아 적절한 num_hidden(2의 거듭제곱)을 리턴.
    규칙: N ≤ 6 → 32, 7~12 → 64, 13↑ → 128
    """
    if num_lines <= 6:
        val = 32
    elif num_lines <= 12:
        val = 64
    else:
        val = 128
    # 혹시 모르는 사용자가 min/max를 바꿔도 안전하게
    val = max(min_ch, min(max_ch, val))
    return val


def calc_num_resblocks(num_lines: int) -> int:
    # 3~6 → 2, 7~12 → 4, 13↑ → 6
    if num_lines <= 6:
        return 2
    elif num_lines <= 12:
        return 4
    else:
        return 6


# PolicyValueNet
N_PLANES = 6
NUM_RESBLOCKS = calc_num_resblocks(NUM_LINES)
NUM_HIDDEN_LAYERS = calc_num_hidden(NUM_LINES)

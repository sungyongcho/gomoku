import numpy as np

NUM_LINES = 19
EMPTY_DOT = "."
PLAYER_X = "X"  # Black
PLAYER_O = "O"  # White

EMPTY_SPACE = 0
PLAYER_1 = 1  # Black
PLAYER_2 = 2  # White
OUT_OF_BOUNDS = 3

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


def opponent_player(player: int) -> int:
    """Return the opponent player number."""
    return PLAYER_2 if player == PLAYER_1 else PLAYER_1


def get_pos(board: np.ndarray, x: int, y: int) -> int:
    """Return the value at column x and row y."""
    return int(board[y, x])


def set_pos(board: np.ndarray, x: int, y: int, value: int) -> None:
    """Set the value at column x and row y."""
    board[y, x] = value


def xy_to_index(x: int, y: int, board_size: int = 19) -> int:
    """Convert (x, y) to a flat index."""
    return x + y * board_size


def index_to_xy(idx: int, board_size: int = 19) -> tuple[int, int]:
    """Convert a flat index to (x, y)."""
    return idx % board_size, idx // board_size


def bulk_index_to_xy(
    indices: np.ndarray, board_width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert flat indices to (x array, y array)."""
    indices = np.asarray(indices, dtype=np.int64)
    x = np.mod(indices, board_width)
    y = indices // board_width
    return x, y


def convert_index_to_coordinates(col: int, row: int, board_size: int = 19) -> str:
    """Convert (col, row) to an 'A1' style string."""
    if not (0 <= col < board_size):
        raise ValueError(f"Column index must be between 0 and {board_size}.")
    if not (0 <= row < board_size):
        raise ValueError(f"Row index must be between 0 and {board_size}.")
    col_char = chr(ord("A") + col)
    return f"{col_char}{row + 1}"


def convert_coordinates_to_xy(coord: str, board_size: int) -> tuple[int, int] | None:
    """Convert 'A1' style text to (x, y) = (col, row); return None if invalid."""
    if coord is None:
        return None
    if not (2 <= len(coord) <= 3):
        return None

    col_letter = coord[0]
    row_number = coord[1:]

    if not col_letter.isalpha() or not row_number.isdigit():
        return None

    x = ord(col_letter.upper()) - ord("A")
    y = int(row_number) - 1

    if not (0 <= x < board_size and 0 <= y < board_size):
        return None

    return x, y


def convert_coordinates_to_index(
    coord: str, board_size: int = 19
) -> tuple[int, int] | None:
    """Convert an 'A1' style coordinate to an (x, y) tuple."""
    if coord is None:
        return None
    if not (2 <= len(coord) <= 3):
        return None

    col_letter = coord[0]
    row_number = coord[1:]

    if not col_letter.isalpha() or not row_number.isdigit():
        return None

    x = ord(col_letter) - ord("A")
    y = int(row_number) - 1

    if not (0 <= x < board_size and 0 <= y < board_size):
        return None

    return x, y

import numpy as np

from gomoku.core.game_config import DIRECTIONS, get_pos, opponent_player

CAPTURE_WINDOW: int = 3


def detect_captured_stones(
    board: np.ndarray, x: int, y: int, player: int, board_size: int
) -> np.ndarray:
    """
    Detect opponent stones captured by placing a stone at ``(x, y)``.

    Parameters
    ----------
    board : numpy.ndarray
        Current board state encoded as integers per cell.
    x : int
        X coordinate of the candidate move.
    y : int
        Y coordinate of the candidate move.
    player : int
        Identifier for the player making the move.
    board_size : int
        Size of the (square) board.

    Returns
    -------
    numpy.ndarray
        Flattened indices of captured opponent stones (row-major). Empty if no
        captures occur.
    """
    # Preallocate for up to two captures in each of eight directions.
    captured_indices = np.empty(16, dtype=np.int16)
    w = 0

    for dx, dy in DIRECTIONS:
        x1, y1 = x + dx, y + dy
        x2, y2 = x + 2 * dx, y + 2 * dy
        x3, y3 = x + 3 * dx, y + 3 * dy
        if not (0 <= x3 < board_size and 0 <= y3 < board_size):
            continue

        opp = opponent_player(player)
        if (
            get_pos(board, x1, y1) == opp
            and get_pos(board, x2, y2) == opp
            and get_pos(board, x3, y3) == player
        ):
            captured_indices[w] = x1 + y1 * board_size
            captured_indices[w + 1] = x2 + y2 * board_size
            w += 2

    if w == 0:
        return captured_indices[:0]
    return captured_indices[:w]

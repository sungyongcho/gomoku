import numpy as np

from gomoku.cpp_ext import renju_cpp


def detect_doublethree(
    board: np.ndarray, x: int, y: int, player: int, board_size: int
) -> bool:
    """Call the C++ double-three detector.

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
    bool
        ``True`` if placing the stone at ``(x, y)`` forms a forbidden
        double-three pattern, otherwise ``False``.
    """
    return renju_cpp.detect_doublethree(board, x, y, player, board_size)

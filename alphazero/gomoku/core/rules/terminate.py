import numpy as np

from gomoku.core.game_config import UNIQUE_DIRECTIONS


def check_local_gomoku(
    pos: np.ndarray, x: int, y: int, player: int, board_size: int, goal_length: int
) -> bool:
    """Check whether placing at ``(x, y)`` completes a winning line.

    Parameters
    ----------
    pos : numpy.ndarray
        Current board state encoded as integers per cell.
    x : int
        X coordinate of the candidate move.
    y : int
        Y coordinate of the candidate move.
    player : int
        Identifier for the player making the move.
    board_size : int
        Size of the (square) board.
    goal_length : int
        Number of contiguous stones required to win (e.g., 5 for Gomoku).

    Returns
    -------
    bool
        ``True`` if the move creates a contiguous line of ``goal_length`` stones
        in any direction, otherwise ``False``.
    """
    for dx, dy in UNIQUE_DIRECTIONS:
        count = 1  # include current stone

        # Check in positive direction
        nx, ny = x + dx, y + dy
        while 0 <= nx < board_size and 0 <= ny < board_size and pos[ny, nx] == player:
            count += 1
            if count == goal_length:
                return True
            nx += dx
            ny += dy

        # Check in negative direction
        nx, ny = x - dx, y - dy
        while 0 <= nx < board_size and 0 <= ny < board_size and pos[ny, nx] == player:
            count += 1
            if count == goal_length:
                return True
            nx -= dx
            ny -= dy

    return False

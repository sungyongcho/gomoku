import numpy as np
from core.board import Board
from core.game_config import NUM_LINES
from core.rules.doublethree import check_doublethree

EMPTY_SPACE = 0


def has_local_five_in_a_row(board: Board, x: int, y: int, player: int) -> bool:
    """Check if (x, y) forms 5-in-a-row using NumPy slicing."""
    array = board.position
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for dx, dy in directions:
        coords_x = np.clip(np.arange(-4, 5) * dx + x, 0, NUM_LINES - 1)
        coords_y = np.clip(np.arange(-4, 5) * dy + y, 0, NUM_LINES - 1)

        line = array[coords_y, coords_x]  # Extract the line
        if len(line) >= 5 and np.any(
            np.convolve(line == player, np.ones(5, dtype=int), mode="valid") == 5
        ):
            return True

    return False


def has_five_in_a_row(board: Board, player: int) -> bool:
    """Check if the player has 5 consecutive stones anywhere using NumPy."""
    array = board.position  # Access the NumPy board directly

    # Convert to boolean mask (True where player stones exist)
    mask = array == player

    # Check rows (horizontal)
    if np.any(np.convolve(mask.ravel(), np.ones(5, dtype=int), mode="valid") == 5):
        return True

    # Check columns (vertical)
    if np.any(np.convolve(mask.T.ravel(), np.ones(5, dtype=int), mode="valid") == 5):
        return True

    # Check diagonals ↘
    for offset in range(-NUM_LINES + 1, NUM_LINES):
        diag = np.diagonal(mask, offset)
        if len(diag) >= 5 and np.any(
            np.convolve(diag, np.ones(5, dtype=int), mode="valid") == 5
        ):
            return True

    # Check diagonals ↙ (flip vertically)
    flipped = np.flipud(mask)
    for offset in range(-NUM_LINES + 1, NUM_LINES):
        diag = np.diagonal(flipped, offset)
        if len(diag) >= 5 and np.any(
            np.convolve(diag, np.ones(5, dtype=int), mode="valid") == 5
        ):
            return True

    return False


def board_is_functionally_full(board: Board) -> bool:
    """Check if the board is full or all moves are forbidden."""
    empty_positions = board.position == EMPTY_SPACE
    if np.any(empty_positions):
        for player in [board.last_player, board.next_player]:
            if np.any(
                [
                    not check_doublethree(board, col, row, player)
                    for col, row in zip(*np.where(empty_positions))
                ]
            ):
                return False
    return True


def is_won_by_score(board: Board, player: int) -> bool:
    """Check if the given player has won by reaching the score goal."""
    target_score = (
        board.last_player_point
        if player == board.last_player
        else board.next_player_point
    )
    return target_score >= board.goal

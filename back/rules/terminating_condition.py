from constants import EMPTY_SPACE, NUM_LINES
from rules.doublethree import check_doublethree
from services.board import Board


def has_local_five_in_a_row(board: Board, x: int, y: int, player: str) -> bool:
    """
    Check if placing 'player' at (x, y) resulted in a 5-in-a-row around (x, y).
    Only scans up to 4 stones in each direction from (x, y).
    """
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        # Count continuous stones in both directions (dx, dy) and (-dx, -dy).
        count = 1  # Include (x, y) itself

        # Forward direction
        nx, ny = x + dx, y + dy
        while (
            0 <= nx < NUM_LINES
            and 0 <= ny < NUM_LINES
            and board.get_value(nx, ny) == player
        ):
            count += 1
            nx += dx
            ny += dy

        # Backward direction
        nx, ny = x - dx, y - dy
        while (
            0 <= nx < NUM_LINES
            and 0 <= ny < NUM_LINES
            and board.get_value(nx, ny) == player
        ):
            count += 1
            nx -= dx
            ny -= dy

        if count >= 5:
            return True
    return False


def has_five_in_a_row(board: Board, player: str) -> bool:
    """Return True if the player has 5 consecutive stones anywhere on the board."""

    for row in range(NUM_LINES):
        for col in range(NUM_LINES):
            if board.get_value(col, row) == player:
                # 1) Check horizontal
                if col + 4 < NUM_LINES:
                    if all(board.get_value(col + i, row) == player for i in range(5)):
                        return True
                # 2) Check vertical
                if row + 4 < NUM_LINES:
                    if all(board.get_value(col, row + i) == player for i in range(5)):
                        return True
                # 3) Check diagonal1 (top-left to bottom-right)
                if col + 4 < NUM_LINES and row + 4 < NUM_LINES:
                    if all(
                        board.get_value(col + i, row + i) == player for i in range(5)
                    ):
                        return True
                # 4) Check diagonal2 (bottom-left to top-right)
                if col + 4 < NUM_LINES and row - 4 >= 0:
                    if all(
                        board.get_value(col + i, row - i) == player for i in range(5)
                    ):
                        return True
    return False


def board_is_functionally_full(board: Board) -> bool:
    """
    Check if neither player has any legal move left:
    - The board might be physically full (no '.'),
    - OR all remaining empty spaces are forbidden by the double-three rule.
    """

    def any_valid_move(board: Board, player: str) -> bool:
        """
        Return True if 'player' has at least one legal move
        (an empty cell that doesn't violate double-three).
        """
        for col in range(NUM_LINES):
            for row in range(NUM_LINES):
                if board.get_value(col, row) == EMPTY_SPACE:
                    # Check if this move is allowed (no double-three)
                    if not check_doublethree(board, col, row, player):
                        return True
        return False

    if any_valid_move(board, board.last_player) or any_valid_move(
        board, board.next_player
    ):
        return False
    return True


def is_won_by_score(board: Board, player: str) -> bool:
    """Check if the given player has won by reaching the score goal."""
    # Determine the target score for the given player
    if player == board.last_player:
        target_score = board.last_player_score
    elif player == board.next_player:
        target_score = board.next_player_score
    else:
        return False  # Invalid player

    # Check if the target score meets the goal
    return target_score >= board.goal

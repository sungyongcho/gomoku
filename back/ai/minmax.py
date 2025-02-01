from typing import List, Tuple

import numpy as np
from constants import NUM_LINES
from rules.capture import capture_opponent
from rules.doublethree import check_doublethree
from rules.terminating_condition import (
    board_is_functionally_full,
    has_five_in_a_row,
    has_local_five_in_a_row,
    is_won_by_score,
)
from services.board import Board

SEARCH_RADIUS = 3

PLAYER_1 = 1
PLAYER_2 = 2
EMPTY_SPACE = 0


def minmax(
    board: Board,
    depth: int,
    is_maximizing: bool,
    alpha: float,
    beta: float,
    player: int,
    opponent: int,
) -> Tuple[int, int, int]:
    """Minimax algorithm with alpha-beta pruning for Gomoku."""
    if depth == 0 or is_terminal(board, player, opponent):
        return evaluate_board(board, player), -1, -1

    best_move = (-1, -1)
    if is_maximizing:
        max_eval = float("-inf")
        for x, y in generate_valid_moves(board, player):
            board.position[y, x] = player  # Direct NumPy indexing (faster)

            if has_local_five_in_a_row(board, x, y, player):
                board.position[y, x] = EMPTY_SPACE  # Undo move
                return (999999, x, y)  # Instant win

            captured_stones = capture_opponent(board, x, y, player)
            board.update_captured_stone(captured_stones)

            eval_score, _, _ = minmax(
                board, depth - 1, False, alpha, beta, player, opponent
            )

            board.position[y, x] = EMPTY_SPACE  # Undo move
            undo_captures(board, captured_stones)

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = (x, y)

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Prune
        return max_eval, best_move[0], best_move[1]
    else:
        min_eval = float("inf")
        for x, y in generate_valid_moves(board, opponent):
            board.position[y, x] = opponent  # Direct NumPy indexing

            if has_local_five_in_a_row(board, x, y, opponent):
                board.position[y, x] = EMPTY_SPACE  # Undo move
                return (-999999, x, y)

            captured_stones = capture_opponent(board, x, y, opponent)
            board.update_captured_stone(captured_stones)

            eval_score, _, _ = minmax(
                board, depth - 1, True, alpha, beta, player, opponent
            )

            board.position[y, x] = EMPTY_SPACE  # Undo move
            undo_captures(board, captured_stones)

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = (x, y)

            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Prune
        return min_eval, best_move[0], best_move[1]


def evaluate_board(board: Board, player: int) -> int:
    """Evaluates the board for minimax scoring."""
    opponent = PLAYER_2 if player == PLAYER_1 else PLAYER_1
    return (
        pattern_score(board, player)
        - pattern_score(board, opponent)
        + detect_capture_threats(board, player, opponent)
    )


def generate_valid_moves(board: Board, player: int) -> List[Tuple[int, int]]:
    """Generate valid moves in a limited search area around existing stones."""
    stones = np.argwhere(board.position != EMPTY_SPACE)

    if stones.size == 0:
        center = NUM_LINES // 2
        radius = 2
        return [
            (col, row)
            for col in range(center - radius, center + radius + 1)
            for row in range(center - radius, center + radius + 1)
            if board.position[row, col] == EMPTY_SPACE
            and not check_doublethree(board, col, row, player)
        ]

    moves = set()
    for x, y in stones:
        for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < NUM_LINES and 0 <= ny < NUM_LINES:
                    if board.position[ny, nx] == EMPTY_SPACE and not check_doublethree(
                        board, nx, ny, player
                    ):
                        moves.add((nx, ny))

    moves = list(moves)
    center = NUM_LINES // 2
    moves.sort(key=lambda move: abs(move[0] - center) + abs(move[1] - center))
    return moves[:20]  # Limit moves for performance


def detect_capture_threats(board: Board, player: int, opponent: int) -> int:
    """Returns a penalty score for potential capture threats."""
    penalty = 0
    for x, y in generate_valid_moves(board, opponent):
        if board.position[y, x] == EMPTY_SPACE:
            board.position[y, x] = opponent
            captured_stones = capture_opponent(board, x, y, opponent)
            board.position[y, x] = EMPTY_SPACE  # Undo move
            undo_captures(board, captured_stones)
            penalty -= sum(5 for stone in captured_stones if stone["stone"] == player)
    return penalty


def pattern_score(board: Board, player: int) -> int:
    """Evaluates the board state based on patterns for a player."""
    points = 0
    opponent = PLAYER_2 if player == PLAYER_1 else PLAYER_1
    three_pattern = np.array([player] * 3, dtype=np.uint8)
    four_pattern = np.array([player] * 4, dtype=np.uint8)
    five_pattern = np.array([player] * 5, dtype=np.uint8)

    for row in range(NUM_LINES):
        points += (
            np.count_nonzero(
                np.convolve(board.position[row, :] == player, np.ones(3), mode="valid")
                == 3
            )
            * 100
        )
        points += (
            np.count_nonzero(
                np.convolve(board.position[row, :] == player, np.ones(4), mode="valid")
                == 4
            )
            * 1000
        )
        points += (
            np.count_nonzero(
                np.convolve(board.position[row, :] == player, np.ones(5), mode="valid")
                == 5
            )
            * 100000
        )

    for col in range(NUM_LINES):
        points += (
            np.count_nonzero(
                np.convolve(board.position[:, col] == player, np.ones(3), mode="valid")
                == 3
            )
            * 100
        )
        points += (
            np.count_nonzero(
                np.convolve(board.position[:, col] == player, np.ones(4), mode="valid")
                == 4
            )
            * 1000
        )
        points += (
            np.count_nonzero(
                np.convolve(board.position[:, col] == player, np.ones(5), mode="valid")
                == 5
            )
            * 100000
        )

    return points


def is_terminal(board: Board, player: int, opponent: int) -> bool:
    """Check if the game has reached a terminal state."""
    return (
        has_five_in_a_row(board, player)
        or has_five_in_a_row(board, opponent)
        or board_is_functionally_full(board)
        or is_won_by_score(board, player)
        or is_won_by_score(board, opponent)
    )


def undo_captures(board: Board, captured_stones: List[dict]) -> None:
    """Undo captures made during Minimax simulation."""
    for stone in captured_stones:
        board.position[stone["y"], stone["x"]] = stone["stone"]

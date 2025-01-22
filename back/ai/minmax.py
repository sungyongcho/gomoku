from typing import List, Tuple

from constants import EMPTY_SPACE, NUM_LINES
from rules.capture import capture_opponent
from rules.doublethree import check_doublethree
from rules.terminating_condition import (
    board_is_functionally_full,
    has_five_in_a_row,
    is_won_by_score,
)
from services.board import Board

SEARCH_RADIUS = 3


def minmax(
    board: Board,
    depth: int,
    is_maximizing: bool,
    alpha: float,
    beta: float,
    player: str,
    opponent: str,
) -> Tuple[int, int, int]:
    """Minimax algorithm with alpha-beta pruning for Gomoku."""
    # Base case: Evaluate the board at the terminal state or max depth
    if depth == 0 or is_terminal(board, player, opponent):
        return evaluate_board(board, player), -1, -1

    best_move = (-1, -1)
    if is_maximizing:
        max_eval = float("-inf")
        for move in generate_valid_moves(board, player):
            x, y = move
            board.set_value(x, y, player)
            captured_stones = capture_opponent(board, x, y, player)
            board.update_captured_stone(captured_stones)
            # Store old score
            if player == board.last_player:
                old_score = board.last_player_score
                board.last_player_score += len(captured_stones)
            else:
                old_score = board.next_player_score
                board.next_player_score += len(captured_stones)

            eval, _, _ = minmax(board, depth - 1, False, alpha, beta, player, opponent)
            board.set_value(x, y, EMPTY_SPACE)  # Undo move
            undo_captures(board, captured_stones)
            if player == board.last_player:
                board.last_player_score = old_score
            else:
                board.next_player_score = old_score

            if eval > max_eval:
                max_eval = eval
                best_move = (x, y)

            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Prune
        return max_eval, best_move[0], best_move[1]
    else:
        min_eval = float("inf")
        for move in generate_valid_moves(board, opponent):
            x, y = move
            board.set_value(x, y, opponent)
            captured_stones = capture_opponent(board, x, y, opponent)
            board.update_captured_stone(captured_stones)
            # Store old score
            if player == board.last_player:
                old_score = board.last_player_score
                board.last_player_score += len(captured_stones)
            else:
                old_score = board.next_player_score
                board.next_player_score += len(captured_stones)

            eval, _, _ = minmax(board, depth - 1, True, alpha, beta, player, opponent)
            board.set_value(x, y, EMPTY_SPACE)  # Undo move
            undo_captures(board, captured_stones)
            if player == board.last_player:
                board.last_player_score = old_score
            else:
                board.next_player_score = old_score

            if eval < min_eval:
                min_eval = eval
                best_move = (x, y)

            beta = min(beta, eval)
            if beta <= alpha:
                break  # Prune
        return min_eval, best_move[0], best_move[1]


def evaluate_board(board: Board, player: str) -> int:
    """
    Evaluate the board for 'player' using simple pattern detection.
    """
    opponent = board.next_player if player == board.last_player else board.last_player
    score = 0

    # Detect patterns for both player and opponent
    score += pattern_score(board, player)
    score -= pattern_score(board, opponent)

    # (Optional) Factor in captures:
    score += (board.last_player_score * 10) if player == board.last_player else 0
    score -= (board.next_player_score * 10) if opponent == board.next_player else 0

    return score


def pattern_score(board: Board, player: str) -> int:
    """
    Assign points for each 3-in-a-row, 4-in-a-row (open or half-open),
    ignoring fully blocked lines. Expand for more detailed patterns.
    """
    points = 0
    size = NUM_LINES

    # Example scoring: +10 per three, +50 per four
    # You can refine with is_open / is_half_open checks
    for row in range(size):
        line = "".join(board.get_row(row))
        points += count_occurrences(line, player * 3) * 10
        points += count_occurrences(line, player * 4) * 50

    for col in range(size):
        line = "".join(board.get_column(col))
        points += count_occurrences(line, player * 3) * 10
        points += count_occurrences(line, player * 4) * 50

    # For diagonals, do similarly
    # e.g., get_diagonal1, get_diagonal2 for each (col, row)
    # be mindful of duplicates or only do it systematically
    # This simplified approach might be enough for a baseline

    return points


def count_occurrences(line: str, pattern: str) -> int:
    """
    Return how many times 'pattern' occurs in 'line'.
    """
    count = 0
    start = 0
    while True:
        idx = line.find(pattern, start)
        if idx == -1:
            break
        count += 1
        start = idx + 1
    return count


def generate_valid_moves(board: Board, player: str) -> List[Tuple[int, int]]:
    """Generate valid moves in a limited search area around existing stones."""
    moves = []
    stone_positions = []

    # 1. Find positions of all stones
    for col in range(NUM_LINES):
        for row in range(NUM_LINES):
            if board.get_value(col, row) != EMPTY_SPACE:
                stone_positions.append((col, row))

    # 2. If no stones found, pick a small region around the center
    if not stone_positions:
        center = NUM_LINES // 2
        radius = 2  # or some small region near center
        for col in range(max(0, center - radius), min(NUM_LINES, center + radius + 1)):
            for row in range(
                max(0, center - radius), min(NUM_LINES, center + radius + 1)
            ):
                if board.get_value(col, row) == EMPTY_SPACE:
                    # Check double-three
                    if not check_doublethree(board, col, row, player):
                        moves.append((col, row))
        return moves

    # 3. Otherwise, search within SEARCH_RADIUS of existing stones
    considered = set()
    for scol, srow in stone_positions:
        for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
                col = scol + dx
                row = srow + dy
                if 0 <= col < NUM_LINES and 0 <= row < NUM_LINES:
                    if board.get_value(col, row) == EMPTY_SPACE:
                        if (col, row) not in considered:
                            if not check_doublethree(board, col, row, player):
                                considered.add((col, row))

    moves = list(considered)

    # Sort by closeness to center (example heuristic)
    center = len(board.get_board()) // 2

    def distance_to_center(move):
        col, row = move
        return abs(col - center) + abs(row - center)

    # Sort moves so closer to center is first
    moves.sort(key=distance_to_center)

    # Keep top 20
    moves = moves[:20]

    # 4. (Optional) Limit the total number of moves by picking top moves
    # e.g., rank moves by closeness to center or other heuristics
    # moves = rank_moves(moves, board, player)[:30]  # keep top 30

    return moves


def is_terminal(board: Board, player: str, opponent: str) -> bool:
    """Check if the board is in a terminal state."""
    # 1. Check if player has 5 in a row
    if has_five_in_a_row(board, player):
        return True
    # 2. Check if opponent has 5 in a row
    if has_five_in_a_row(board, opponent):
        return True
    # 3. Check if the board is completely filled
    if board_is_functionally_full(board):
        return True
    # 4. Check if the capture score is met for player
    if is_won_by_score(board, player):
        return True
    # 5. Check if the capture score is met for opponent
    if is_won_by_score(board, opponent):
        return True

    return False


def undo_captures(board: Board, captured_stones: List[dict]) -> None:
    """Undo captures made during the Minimax simulation."""
    for stone in captured_stones:
        print(stone)
        board.set_value(stone["x"], stone["y"], stone["stone"])

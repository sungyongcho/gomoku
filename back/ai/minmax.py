from typing import List, Tuple

from constants import EMPTY_SPACE, NUM_LINES, PLAYER_1, PLAYER_2
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

            # Immediate Win?
            if has_local_five_in_a_row(board, x, y, player):
                board.set_value(x, y, EMPTY_SPACE)
                print("testing", x, y)
                return (999999, x, y)

            captured_stones = capture_opponent(board, x, y, player)
            if len(captured_stones) > 0:
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
            if len(captured_stones) > 0:
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
            # If the MINIMIZING side has 5 in a row => from the maximizing perspective = -999999
            if has_local_five_in_a_row(board, x, y, player):
                board.set_value(x, y, EMPTY_SPACE)
                return (-999999, x, y)
            captured_stones = capture_opponent(board, x, y, opponent)
            if len(captured_stones) > 0:
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
            if len(captured_stones) > 0:
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
    opponent = board.next_player if player == board.last_player else board.last_player

    # Pattern scoring
    base_score = pattern_score(board, player) - pattern_score(board, opponent)

    # Capture scoring
    base_score += (board.last_player_score * 10) if player == board.last_player else 0
    base_score -= (board.next_player_score * 10) if opponent == board.next_player else 0

    # Capture threat penalty
    capture_threat_penalty = detect_capture_threats(board, player, opponent)

    return base_score + capture_threat_penalty


def detect_capture_threats(board: Board, player: str, opponent: str) -> int:
    """
    Returns a negative penalty if the opponent can capture 'player''s stones
    on their next move.

    Strategy:
      1. Generate a small set of empty cells to consider (like a 2-3 cell radius).
      2. For each cell, place 'opponent' stone, call capture_opponent.
      3. If any stones from 'player' would be captured, apply a penalty.
      4. Undo and move on.

    Final penalty is e.g., -5 times the number of your stones that can be captured
    by the opponent in a single move.
    """
    threat_penalty = 0

    # Gather potential moves for the opponent
    # We can reuse your "generate_valid_moves" or do a smaller, local approach
    possible_moves = generate_threat_moves(board, opponent, radius=2)

    for col, row in possible_moves:
        # Temporarily place the opponent's stone
        if board.get_value(col, row) == EMPTY_SPACE:
            board.set_value(col, row, opponent)
            captured_stones = capture_opponent(board, col, row, opponent)

            # Undo
            board.set_value(col, row, EMPTY_SPACE)

            # Only restore if your code removes them immediately in capture_opponent
            undo_captures(board, captured_stones)

            # If any captured stone belongs to 'player', penalize
            # e.g., -5 for each stone that would be captured
            penalty_for_this_move = 0
            for stone in captured_stones:
                if stone["stone"] == player:
                    penalty_for_this_move -= 5

            threat_penalty += penalty_for_this_move

    # Summation of all possible capture threats
    return threat_penalty


def generate_threat_moves(board: Board, player: str, radius=2) -> List[Tuple[int, int]]:
    """
    Return a limited set of empty cells near existing stones,
    to check for potential capture threats.
    """
    size = len(board.get_board())
    stone_positions = []
    moves = set()

    # Find all stones
    for c in range(size):
        for r in range(size):
            if board.get_value(c, r) != EMPTY_SPACE:
                stone_positions.append((c, r))

    # If no stones, no threat
    if not stone_positions:
        return []

    for sc, sr in stone_positions:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nc, nr = sc + dx, sr + dy
                if 0 <= nc < size and 0 <= nr < size:
                    if board.get_value(nc, nr) == EMPTY_SPACE:
                        moves.add((nc, nr))

    return list(moves)


def pattern_score(board: Board, player: str) -> int:
    points = 0
    size = NUM_LINES
    opponent = PLAYER_2 if player == PLAYER_1 else PLAYER_1

    # We'll do it for "three" and "four" patterns
    three_pattern = player * 3
    four_pattern = player * 4
    five_pattern = player * 5

    for row in range(size):
        row_line = "".join(board.get_row(row))
        points += (
            count_occurrences_with_context(row_line, three_pattern, player, opponent)
            * 100
        )
        points += (
            count_occurrences_with_context(row_line, four_pattern, player, opponent)
            * 1000
        )
        points += (
            count_occurrences_with_context(row_line, five_pattern, player, opponent)
            * 100000
        )

    for col in range(size):
        col_line = "".join(board.get_column(col))
        points += (
            count_occurrences_with_context(col_line, three_pattern, player, opponent)
            * 100
        )
        points += (
            count_occurrences_with_context(col_line, four_pattern, player, opponent)
            * 1000
        )
        points += (
            count_occurrences_with_context(col_line, five_pattern, player, opponent)
            * 100000
        )

    # Diagonals (downward and upward)
    for diag in board.get_all_downward_diagonals():
        diag_line = "".join(diag)
        points += (
            count_occurrences_with_context(diag_line, three_pattern, player, opponent)
            * 100
        )
        points += (
            count_occurrences_with_context(diag_line, four_pattern, player, opponent)
            * 1000
        )
        points += (
            count_occurrences_with_context(diag_line, five_pattern, player, opponent)
            * 100000
        )

    for diag in board.get_all_upward_diagonals():
        diag_line = "".join(diag)
        points += (
            count_occurrences_with_context(diag_line, three_pattern, player, opponent)
            * 100
        )
        points += (
            count_occurrences_with_context(diag_line, four_pattern, player, opponent)
            * 1000
        )
        points += (
            count_occurrences_with_context(diag_line, five_pattern, player, opponent)
            * 100000
        )

    return points


def count_occurrences_with_context(
    line: str, pattern: str, player: str, opponent: str
) -> int:
    """
    For each occurrence of 'pattern' in 'line', check the surrounding context
    to decide if it's open/half-open/blocked. Return a cumulative score.

    Example scoring (tweak as you like):
      - fully open => +3
      - half open  => +2
      - blocked    => +1 (or 0 if you want to ignore fully-blocked patterns)
    """
    total_score = 0
    start = 0

    while True:
        idx = line.find(pattern, start)
        if idx == -1:
            break

        left_idx = idx - 1
        right_idx = idx + len(pattern)

        # Get left/right chars, treat out-of-bounds as 'opponent' or a block
        left_char = line[left_idx] if 0 <= left_idx < len(line) else opponent
        right_char = line[right_idx] if 0 <= right_idx < len(line) else opponent

        # Decide if open, half-open, or blocked
        if left_char == "." and right_char == ".":
            # fully open
            total_score += 3
        elif (left_char == "." and right_char != "." and right_char != player) or (
            right_char == "." and left_char != "." and left_char != player
        ):
            # half open => +2
            total_score += 2
        # else:
        #     # fully blocked => +1 or 0
        #     total_score += 1

        start = idx + 1  # move past this occurrence

    return total_score


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
        board.set_value(stone["x"], stone["y"], stone["stone"])

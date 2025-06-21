from __future__ import annotations

from typing import Tuple

import numpy as np

from game_config import (
    CAPTURE_GOAL,
    EMPTY_SPACE,
    NUM_LINES,
    PLAYER_1,
    PLAYER_2,
    get_pos,
    opponent_player,
    set_pos,
)
from rules.capture import detect_captured_stones
from rules.doublethree import detect_doublethree
from rules.terminate import check_local_gomoku


class Gomoku:
    def __init__(self):
        self.row_count = NUM_LINES
        self.col_count = NUM_LINES
        self.last_captures = []
        self.p1_pts: int = 0
        self.p2_pts: int = 0
        self.action_size = self.row_count * self.col_count
        self.enable_doublethree = True
        self.enable_capture = True

    def get_initial_state(self) -> np.ndarray:
        return np.zeros((self.row_count, self.col_count))

    def get_next_state(
        self, state: np.ndarray, action: Tuple[int, int], player: int
    ) -> np.ndarray:
        x, y = action
        set_pos(state, x, y, player)

        if self.enable_capture:
            captures = detect_captured_stones(state, x, y, player)
            if len(captures) > 0:
                self.last_captures = captures
                for s in captures:
                    set_pos(state, s["x"], s["y"], EMPTY_SPACE)
                if player == PLAYER_1:
                    self.p1_pts += len(captures) // 2
                else:
                    self.p2_pts += len(captures) // 2

        return state

    def get_legal_moves(self, state: np.ndarray, player: int) -> list[str]:
        """Returns a list of valid moves in [A1, B5, ...] format."""
        size = state.shape[0]
        moves = []

        for y in range(size):
            for x in range(size):
                if get_pos(state, x, y) == EMPTY_SPACE:
                    if self.enable_doublethree and not detect_doublethree(
                        state, x, y, player
                    ):
                        col = chr(ord("A") + x)
                        row = str(y + 1)
                        moves.append(f"{col}{row}")

        return moves

    def check_win(self, state: np.ndarray, action: Tuple[int, int]) -> bool:
        if action is None:
            return False
        x, y = action
        player = get_pos(state, x, y)

        if check_local_gomoku(state, x, y, player):
            return True
        if player == PLAYER_1 and self.p1_pts >= CAPTURE_GOAL:
            return True
        if player == PLAYER_2 and self.p2_pts >= CAPTURE_GOAL:
            return True

        return False

    def get_value_and_terminated(self, state: np.ndarray, action: Tuple[int, int]):
        if self.check_win(state, action):
            return 1, True
        if len(self.get_legal_moves(state, PLAYER_1)) == 0 and len(
            self.get_legal_moves(state, PLAYER_2)
        ):
            return 0, True
        return 0, False

    def print_board(self, state: np.ndarray) -> None:
        """Prints the board with column letters (A-T) and row numbers (1-19)."""
        size = len(state)
        column_labels = " ".join(chr(ord("A") + i) for i in range(size))
        print("   " + column_labels)
        for i, row in enumerate(state):
            row_label = f"{i + 1:>2}"  # Right-align single-digit numbers
            row_str = " ".join(str(int(cell)) for cell in row)
            print(f"{row_label} {row_str}")


def convert_coordinates_to_index(coord: str) -> Tuple[int, int] | None:
    """
    Converts a coordinate string like 'A1' or 'T19' to (x, y) index Tuple.
    Returns None if the format is invalid.
    """
    if not (2 <= len(coord) and len(coord) <= 3):
        return None

    col_letter = coord[0]
    row_number = coord[1:]

    if not col_letter.isalpha() or not row_number.isdigit():
        return None

    x = ord(col_letter) - ord("A")
    y = int(row_number) - 1
    print(x, y)

    if x < 0 or x > 19:
        return None

    if y < 0 or y > 19:
        return None

    return x, y


gomoku = Gomoku()

player = PLAYER_1

state = gomoku.get_initial_state()

set_pos(state, 7, 7, PLAYER_1)
set_pos(state, 8, 7, PLAYER_2)
set_pos(state, 9, 7, PLAYER_2)

set_pos(state, 10, 4, PLAYER_1)
set_pos(state, 10, 5, PLAYER_2)
set_pos(state, 10, 6, PLAYER_2)

set_pos(state, 11, 7, PLAYER_2)
set_pos(state, 12, 7, PLAYER_2)
set_pos(state, 13, 7, PLAYER_1)

set_pos(state, 10, 8, PLAYER_2)
set_pos(state, 10, 9, PLAYER_2)
set_pos(state, 10, 10, PLAYER_1)


while True:
    gomoku.print_board(state)
    valid_moves = gomoku.get_legal_moves(state, player)
    print("valid_moves:", valid_moves)

    action_str = input(f"{player} (e.g. A1): ").strip().upper()
    if action_str not in valid_moves:
        print("Action not valid -- invalid string")
        continue

    action_coord = convert_coordinates_to_index(action_str)
    if action_coord is None:
        print("Action not valid -- 2 ")
        continue

    state = gomoku.get_next_state(state, action_coord, player)

    # gomoku.print_board(state)
    value, is_terminal = gomoku.get_value_and_terminated(state, action_coord)

    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break

    player = opponent_player(player)

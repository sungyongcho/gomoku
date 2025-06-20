from __future__ import annotations

import numpy as np

from game_config import NUM_LINES, PLAYER_1, get_pos, opponent_player, set_pos
from rules.terminate import check_local_gomoku


class Gomoku:
    def __init__(self):
        self.row_count = NUM_LINES
        self.col_count = NUM_LINES
        # self.last_x: int | None = None
        # self.last_y: int | None = None
        # self.last_player: int | None = None
        self.action_size = self.row_count * self.col_count

    def get_initial_state(self) -> np.ndarray:
        return np.zeros((self.row_count, self.col_count))

    def get_next_state(
        self, state: np.ndarray, action: tuple[int, int], player: int
    ) -> np.ndarray:
        x, y = action
        # self.last_player = player
        # self.last_x = x
        # self.last_y = y
        set_pos(state, x, y, player)
        return state

    def get_valid_moves(self, state: np.ndarray) -> list[str]:
        """Returns a list of valid moves in [A1, B5, ...] format."""
        size = state.shape[0]
        moves = []

        for y in range(size):
            for x in range(size):
                if state[y, x] == 0:
                    col = chr(ord("A") + x)
                    row = str(y + 1)
                    moves.append(f"{col}{row}")

        return moves

    def check_win(self, state: np.ndarray, action: tuple[int, int]) -> bool:
        if action is None:
            return False
        x, y = action
        player = get_pos(state, x, y)

        if check_local_gomoku(state, x, y, player):
            return True

        return False

    def get_value_and_terminated(self, state: np.ndarray, action: tuple[int, int]):
        if self.check_win(state, action):
            return 1, True
        if len(self.get_valid_moves(state)) == 0:
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


def convert_coordinates_to_index(coord: str) -> tuple[int, int] | None:
    """
    Converts a coordinate string like 'A1' or 'T19' to (x, y) index tuple.
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

while True:
    gomoku.print_board(state)
    valid_moves = gomoku.get_valid_moves(state)
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

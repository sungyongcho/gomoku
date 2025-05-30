from typing import List

import numpy as np
from core.game_config import (
    EMPTY_DOT,
    EMPTY_SPACE,
    PLAYER_1,
    PLAYER_2,
    PLAYER_X,
)


class Board:
    def __init__(
        self,
        board_data: dict,
    ) -> None:
        """Initialize the board from a provided game state dictionary."""
        self.goal: int = board_data["goal"]
        self.last_player: int = (
            PLAYER_1 if board_data["lastPlay"]["stone"] == PLAYER_X else PLAYER_2
        )
        self.next_player: int = (
            PLAYER_1 if board_data["nextPlayer"] == PLAYER_X else PLAYER_2
        )
        self._last_player_point: int = next(
            s["score"] for s in board_data["scores"] if s["player"] == PLAYER_X
        )
        self._next_player_point = next(
            s["score"] for s in board_data["scores"] if s["player"] == PLAYER_X
        )

        # Convert board from list of strings to NumPy array
        self.position: np.array = np.array(
            [
                [
                    EMPTY_SPACE
                    if cell == EMPTY_DOT
                    else (PLAYER_1 if cell == PLAYER_X else PLAYER_2)
                    for cell in row
                ]
                for row in board_data["board"]
            ],
            dtype=np.uint8,
        )
        self.enable_capture: bool = board_data["enableCapture"]
        self.enable_doublethree: bool = board_data["enableDoubleThreeRestriction"]
        last_x = board_data["lastPlay"]["coordinate"].get("x")
        last_y = board_data["lastPlay"]["coordinate"].get("y")

        if last_x is not None and last_y is not None:
            self.last_x: int | None = last_x
            self.last_y: int | None = last_y

    def __getitem__(self, indices: tuple[int, int]) -> int:
        """Get the value at a specific column and row."""
        return self.position[indices]

    def __setitem__(self, indices: tuple[int, int], value: int) -> None:
        """Set the value at a specific column and row."""
        self.position[indices] = value

    def get_board(self) -> np.ndarray:
        """Get the current board state."""
        return self.position

    def reset_board(self) -> None:
        """Resets the board to an empty state."""
        self.position.fill(EMPTY_SPACE)
        self._last_player_point = 0
        self._next_player_point = 0
        self.last_x = None
        self.last_y = None

    def get_value(self, col: int, row: int) -> int:
        """Get the value at a specific column and row."""
        return self.position[row, col]

    def switch_turn(self):
        self.last_player, self.next_player = self.next_player, self.last_player
        self._last_player_point, self._next_player_point = (
            self._next_player_point,
            self._last_player_point,
        )

    def set_value(self, col: int, row: int, value: int) -> None:
        """Set the value at a specific column and row."""
        self.position[row, col] = value
        # if value != EMPTY_SPACE:
        #     self.update_last_move(col, row)

    def update_last_move(self, x: int, y: int) -> None:
        self.last_x = x
        self.last_y = y

    def get_row(self, row: int) -> np.ndarray:
        """Return a specific row."""
        return self.position[row, :]

    def get_column(self, col: int) -> np.ndarray:
        """Return a specific column."""
        return self.position[:, col]

    def update_captured_stone(self, captured_stones: List[dict]) -> None:
        """Removes captured stones from the board."""
        for captured in captured_stones:
            self.position[captured["y"], captured["x"]] = EMPTY_SPACE

    def print_board(self) -> None:
        """Prints the board with column letters (A-T) and row numbers (1-19)."""
        size = len(self.position)
        column_labels = " ".join(chr(ord("A") + i) for i in range(size))
        print("   " + column_labels)
        for i, row in enumerate(self.position):
            row_label = f"{i + 1:>2}"  # Right-align single-digit numbers
            row_str = " ".join(map(str, row))
            print(f"{row_label} {row_str}")

    @staticmethod
    def convert_index_to_coordinates(col: int, row: int) -> str:
        if not (0 <= col < 19):
            raise ValueError("Column index must be between 0 and 18.")
        if not (0 <= row < 19):
            raise ValueError("Row index must be between 0 and 18.")
        col_char = chr(ord("A") + col)
        return f"{col_char}{row + 1}"

    @property
    def current_player(self) -> int:
        return self.next_player

    @property
    def next_player_point(self) -> int:
        return self._next_player_point

    @property
    def last_player_point(self) -> int:
        return self._last_player_point

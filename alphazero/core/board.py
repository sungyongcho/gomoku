from typing import List

import numpy as np
from core.game_config import NUM_LINES

PLAYER_1 = 1
PLAYER_2 = 2
EMPTY_SPACE = 0


class Board:
    def __init__(
        self,
        board_data: dict,
    ) -> None:
        """Initialize the board from a provided game state dictionary."""
        self.goal = board_data["goal"]
        self.last_player = (
            PLAYER_1 if board_data["lastPlay"]["stone"] == "X" else PLAYER_2
        )
        self.next_player = PLAYER_1 if board_data["nextPlayer"] == "X" else PLAYER_2
        self.last_player_score = next(
            s["score"] for s in board_data["scores"] if s["player"] == "X"
        )
        self.next_player_score = next(
            s["score"] for s in board_data["scores"] if s["player"] == "O"
        )

        # Convert board from list of strings to NumPy array
        self.position = np.array(
            [
                [
                    EMPTY_SPACE
                    if cell == "."
                    else (PLAYER_1 if cell == "X" else PLAYER_2)
                    for cell in row
                ]
                for row in board_data["board"]
            ],
            dtype=np.uint8,
        )

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

    def get_value(self, col: int, row: int) -> int:
        """Get the value at a specific column and row."""
        return self.position[row, col]

    def set_value(self, col: int, row: int, value: int) -> None:
        """Set the value at a specific column and row."""
        self.position[row, col] = value

    def get_row(self, row: int) -> np.ndarray:
        """Return a specific row."""
        return self.position[row, :]

    def get_column(self, col: int) -> np.ndarray:
        """Return a specific column."""
        return self.position[:, col]

    def get_all_downward_diagonals(self) -> List[np.ndarray]:
        """Return all downward (\) diagonals as NumPy arrays."""
        return [self.position.diagonal(i) for i in range(-NUM_LINES + 1, NUM_LINES)]

    def get_all_upward_diagonals(self) -> List[np.ndarray]:
        """Return all upward (/) diagonals as NumPy arrays."""
        flipped_board = np.fliplr(self.position)
        return [flipped_board.diagonal(i) for i in range(-NUM_LINES + 1, NUM_LINES)]

    def update_captured_stone(self, captured_stones: List[dict]) -> None:
        """Removes captured stones from the board."""
        for captured in captured_stones:
            self.position[captured["y"], captured["x"]] = EMPTY_SPACE

    def convert_board_for_print(self) -> str:
        """Converts the board to a human-readable string."""
        return "\n".join(" ".join(map(str, row)) for row in self.position)

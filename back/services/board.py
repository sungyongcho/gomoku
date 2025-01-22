from typing import List

from constants import EMPTY_SPACE, NUM_LINES


class Column:
    def __init__(self, column: List[str]) -> None:
        """Initialize a column."""
        self.column = column

    def __getitem__(self, row: int) -> str:
        """Get value at (col, row)."""
        return self.column[row]

    def __setitem__(self, row: int, value: str) -> None:
        """Set value at (col, row)."""
        self.column[row] = value

    def __repr__(self) -> str:
        """String representation of the column."""
        return repr(self.column)


class Board:
    def __init__(
        self,
        board: List[List[str]] = None,
        last_player: str = "X",  # this is not correct, just for the example
        next_player: str = "O",
        last_player_score: int = 0,
        next_player_score: int = 0,
        goal: int = 10,
    ) -> None:
        """Initialize the board."""
        self.position = board or [[EMPTY_SPACE] * NUM_LINES for _ in range(NUM_LINES)]
        self.last_player = last_player
        self.next_player = next_player
        self.last_player_score = last_player_score
        self.next_player_score = next_player_score
        self.goal = goal

    def __getitem__(self, col: int) -> Column:
        """Get a column to support [][] access."""
        return Column([row[col] for row in self.position])

    def set_board(self, board: List[List[str]]) -> None:
        """Set the entire board."""
        self.position = board

    def get_board(self) -> List[List[str]]:
        """Get the current board state."""
        return self.position

    def reset_board(self) -> None:
        """Resets the board to an empty state."""
        self.position = [[EMPTY_SPACE] * NUM_LINES for _ in range(NUM_LINES)]

    def get_value(self, col: int, row: int) -> str:
        """Get the value at a specific column and row."""
        return self.position[row][col]

    def set_value(self, col: int, row: int, value: str) -> None:
        """Set the value at a specific column and row."""
        self.position[row][col] = value

    def get_row(self, row: int) -> List[str]:
        """Return a specific row."""
        return self.position[row]

    def get_column(self, col: int) -> List[str]:
        """Return a specific column."""
        return [row[col] for row in self.position]

    def get_diagonal1(self, col: int, row: int) -> List[str]:
        """Return the top-left to bottom-right diagonal passing through (col, row)."""
        diag = []
        size = len(self.position)
        for i in range(-min(col, row), size - max(col, row)):
            diag.append(self.get_value(col + i, row + i))
        return diag

    def get_diagonal2(self, col: int, row: int) -> List[str]:
        """Return the bottom-left to top-right diagonal passing through (col, row)."""
        diag = []
        size = len(self.position)
        for i in range(-min(col, size - row - 1), min(size - col, row + 1)):
            diag.append(self.get_value(col + i, row - i))
        return diag

    def update_captured_stone(self, captured_stones: list) -> None:
        for captured in captured_stones:
            self.set_value(captured["x"], captured["y"], EMPTY_SPACE)

    def convert_board_for_print(self) -> str:
        """Converts the board to a human-readable string."""
        board_to_print = ""
        for row in self.position:
            board_to_print += "".join(cell for cell in row) + "\n"
        return board_to_print

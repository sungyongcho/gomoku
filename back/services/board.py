from typing import List

from constants import EMPTY_SPACE, NUM_LINES


class Column:
    def __init__(self, column):
        self.column = column

    def __getitem__(self, row: int):
        """Get value at (col, row)"""
        return self.column[row]

    def __setitem__(self, row: int, value):
        """Set value at (col, row)"""
        self.column[row] = value


class Board:
    def __init__(self, board: List[List[str]] = None) -> None:
        # define board position
        self.position = board or [[EMPTY_SPACE] * NUM_LINES for _ in range(NUM_LINES)]
        # create a copy of previous board state if available
        # if board is not None:
        #     self.__dict__ = deepcopy(board.__dict__)

    def __getitem__(self, col: int) -> Column:
        """Get a column to support [][] access"""
        return Column([row[col] for row in self.position])

    def set_board(self, board: List[List[str]]) -> None:
        self.position = board

    def reset_board(self):
        """Resets the board to an empty state."""
        self.position = [[EMPTY_SPACE] * NUM_LINES for _ in range(NUM_LINES)]

    def get_value(self, col: int, row: int) -> str:
        return self.position[row][col]

    def set_value(self, col: int, row: int, value: str) -> str:
        self.position[row][col] = value
        return value

    def convert_board_for_print(
        self,
    ):
        """Converts the board to a human-readable string."""
        board_to_print = ""
        for row in self.position:
            board_to_print += "".join(cell for cell in row) + "\n"
        return board_to_print

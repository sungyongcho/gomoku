import time
from typing import List

from constants import EMPTY_SPACE
from rules.capture import capture_opponent
from rules.doublethree import check_doublethree
from services.board import Board


class Gomoku:
    def __init__(self, board_size=19):
        self.board_size = board_size
        self.board = Board()
        self.capture_goal = 0
        # self.history = []
        self.last_player = ""
        self.next_player = ""
        self.last_player_capture = 0
        self.next_player_capture = 0
        self.captured_stones = []
        # # for testing - capture
        # self.board.set_value(2, 1, "O")
        # self.board.set_value(3, 1, "O")
        # self.board.set_value(4, 1, "X")

        # self.board.set_value(2, 2, "O")
        # self.board.set_value(3, 3, "O")
        # self.board.set_value(4, 4, "X")

        # self.board.set_value(1, 2, "O")
        # self.board.set_value(1, 3, "O")
        # self.board.set_value(1, 4, "X")
        # for testing - doublethree
        # self.board.set_value(0, 1, "X")
        # self.board.set_value(2, 1, "X")
        # self.board.set_value(3, 1, "X")

    def set_game(
        self,
        board: List[List[str]],
        goal: int,
        last_player: str,
        next_player: str,
    ) -> None:
        self.board.set_board(board)
        self.capture_goal = goal
        self.last_player = last_player
        self.next_player = next_player

    def print_board(self) -> str:
        return self.board.convert_board_for_print()

    def get_board(self) -> List[List[str]]:
        return self.board.get_board()

    def get_scores(self):
        return [
            {"player": self.last_player, "score": self.last_player_capture},
            {"player": self.next_player, "score": self.next_player_capture},
        ]

    def get_captured_stones(self):
        return self.captured_stones

    def reset_board(self) -> None:
        self.board.reset_board()
        # self.history = []
        self.last_player_capture = 0
        self.next_player_capture = 0

    def update_board(self, x: int, y: int, player: str) -> bool:
        captured_stones = capture_opponent(self.board, x, y, player)
        if captured_stones:
            self.captured_stones = captured_stones
            self.remove_captured_stone(captured_stones)
        elif self.is_doublethree(self.board, x, y, player):
            return False
        return True

    def remove_captured_stone(self, captured_stones: list) -> None:
        for captured in captured_stones:
            self.board.set_value(captured["x"], captured["y"], EMPTY_SPACE)
            if captured["stone"] == self.next_player:
                self.last_player_capture += 1
            else:
                self.next_player_capture += 1
            # self.record_history(
            #     pair[0],
            #     pair[1],
            #     PLAYER_1 if player == PLAYER_2 else PLAYER_2,
            #     "capture",
            # )
        print(
            f"Capture score - {self.last_player}: {self.last_player_capture}, {self.next_player}: {self.next_player_capture}"
        )

    def is_doublethree(self, board: Board, x: int, y: int, player: str) -> bool:
        return check_doublethree(board, x, y, player)

    def place_stone(self, x: int, y: int, player: str) -> bool:
        if (
            0 <= x < self.board_size
            and 0 <= y < self.board_size
            and self.board[x][y] == "."
        ):
            self.board.set_value(x, y, player)
            # self.record_history(x, y, player, "place")
            return True
        return False

    def play_next(self):
        start_time = time.time()  # Record the start time
        self.place_stone(1, 1, "O")
        end_time = time.time()  # Record the end time
        elapsed_time_ms = (end_time - start_time) * 1000  # Convert to ms
        print(f"Update processed in {elapsed_time_ms:.3f} ms")

        return {"coordinate": {"x": 1, "y": 1}, "stone": "O"}

    # def record_history(self, x: int, y: int, player: str, type: str) -> None:
    #     self.history.append({"x": x, "y": y, "player": player, "type": type})

    # def print_history(self) -> None:
    #     print(self.history)


# def play_next():
#     start_time = time.time()  # Record the start time

#     last_move = history[len(history) - 1]
#     x, y = last_move["x"], last_move["y"]
#     # place_stone(x + 1, y + 1, "O")
#     end_time = time.time()  # Record the end time
#     elapsed_time_ms = (end_time - start_time) * 1000  # Convert to ms
#     print(f"Update processed in {elapsed_time_ms:.3f} ms")
#     pass

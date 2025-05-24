import time
from typing import List

from constants import EMPTY_SPACE
from rules.capture import capture_opponent
from rules.doublethree import check_doublethree
from services.board import Board


class Gomoku:
    def __init__(self, board_size=19):
        self.board_size = board_size
        self.board = None
        self.goal = 0
        # self.history = []
        self.last_player = ""
        self.next_player = ""
        self.last_player_score = 0
        self.next_player_score = 0
        self.captured_stones = []

    # def set_game(
    #     self,
    #     board: List[List[str]],
    #     last_player: str,
    #     next_player: str,
    #     scores: List[dict],
    #     goal: int,
    # ) -> None:
    #     self.goal = goal * 2
    #     self.last_player = last_player
    #     self.next_player = next_player

    #     # Convert scores list to a dictionary
    #     scores_dict = {score["player"]: score["score"] for score in scores}

    #     # Access the scores using last_player and next_player
    #     self.last_player_score = scores_dict.get(last_player, 0) * 2
    #     self.next_player_score = scores_dict.get(next_player, 0) * 2

    #     self.board = Board(
    #         board,
    #         last_player,
    #         next_player,
    #         self.last_player_score,
    #         self.next_player_score,
    #         self.goal,
    #     )

    def set_game(self, board_data: dict) -> None:
        self.board = Board(board_data)

    def print_board(self) -> str:
        return self.board.convert_board_for_print()

    def get_board(self) -> List[List[str]]:
        return self.board.get_board()

    def get_scores(self):
        return [
            {"player": self.last_player, "score": int(self.last_player_score / 2)},
            {"player": self.next_player, "score": int(self.next_player_score / 2)},
        ]

    def get_captured_stones(self):
        return self.captured_stones

    def reset_board(self) -> None:
        self.board.reset_board()
        # self.history = []
        self.last_player_score = 0
        self.next_player_score = 0

    def update_board(self, x: int, y: int, player: str) -> bool:
        captured_stones = capture_opponent(self.board, x, y, player)
        if captured_stones:
            # self.captured_stones = captured_stones
            self.remove_captured_stone(captured_stones)
        elif self.is_doublethree(self.board, x, y, player):
            return False
        return True

    def remove_captured_stone(self, captured_stones: list) -> None:
        for captured in captured_stones:
            self.board.set_value(captured["x"], captured["y"], EMPTY_SPACE)
            # if captured["stone"] == self.next_player:
            #     self.last_player_score += 1
            # else:
            #     self.next_player_score += 1
            # self.record_history(
            #     pair[0],
            #     pair[1],
            #     PLAYER_1 if player == PLAYER_2 else PLAYER_2,
            #     "capture",
            # )
        print(
            f"Capture score - {self.last_player}: {self.last_player_score}, {self.next_player}: {self.next_player_score}"
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

    def play_next_minmax(self):
        """
        Determine and play the next move for the AI using the Minimax algorithm.
        """
        start_time = time.time()

        # Call the Minimax function to get the best move
        score, col, row = minmax(
            board=self.board,
            depth=3,  # Adjust depth for performance vs. accuracy
            is_maximizing=True,
            alpha=float("-inf"),
            beta=float("inf"),
            player=self.board.next_player,
            opponent=self.board.last_player,
        )

        # If a valid move is found, place the stone
        if col != -1 and row != -1:
            captured_stones = capture_opponent(
                self.board, col, row, self.board.next_player
            )
            if captured_stones:
                self.captured_stones = captured_stones
                self.remove_captured_stone(captured_stones)
            if self.place_stone(col, row, self.next_player):
                print(f"AI placed stone at ({col}, {row}) with score {score}")

        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"AI chose move ({col}, {row}) in {elapsed_time_ms:.3f} ms")

        # Return the move details
        return {"coordinate": {"col": col, "row": row}, "stone": self.next_player}

        # testing
        print(self.next_player, flush=True)
        self.place_stone(1, 1, self.next_player)
        return {"coordinate": {"x": 1, "y": 1}, "stone": self.next_player}

    # def record_history(self, x: int, y: int, player: str, type: str) -> None:
    #     self.history.append({"x": x, "y": y, "player": player, "type": type})

    # def print_history(self) -> None:
    #     print(self.history)

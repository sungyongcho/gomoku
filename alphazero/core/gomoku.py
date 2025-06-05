from core.board import Board
from core.game_config import DRAW
from core.rules.terminate import (
    board_is_functionally_full,
    check_local_gomoku,
    is_won_by_score,
)


class Gomoku:
    def __init__(self, board: Board | None = None):
        self.board: Board = board or Board.empty()
        self.turn: int = 0
        self.history: list[tuple[int, int, int]] = []
        self.winner: int | None = None
        self.last_captures: list[dict] = []

    @property
    def current_player(self):
        return self.board.next_player

    def play_move(self, x: int, y: int) -> bool:
        if not self.board.is_legal_move(x, y, self.current_player):
            return False
        self.board.apply_move(x, y, self.current_player)
        self.history.append((x, y, self.current_player))
        self._check_terminal()
        self.turn += 1
        # 턴 스왑
        self.board.last_player, self.board.next_player = (
            self.board.next_player,
            self.board.last_player,
        )
        self.board.last_pts, self.board.next_pts = (
            self.board.next_pts,
            self.board.last_pts,
        )
        return True

    def _check_terminal(self) -> None:
        # 5목·점수·가득참 체크 → self.winner 갱신
        if self.board.last_x is None or self.board.last_y is None:
            return

        if check_local_gomoku(
            self.board, self.board.last_x, self.board.last_y, self.current_player
        ) or is_won_by_score(self.board, self.current_player):
            self.winner = self.current_player

        elif board_is_functionally_full(self.board):
            self.winner = DRAW

    def set_board(self, data: dict):
        self.board.update_from_dict(data)
        self.turn = data.get("turn", 0)
        self.history.clear()
        self.winner = None
        self.last_captures = []

    # 디버그 전체 출력
    def debug_print(self):
        print(
            f"Turn {self.turn}  Player {self.board.next_player}"
            f"  Score B:{self.board.last_pts} W:{self.board.next_pts}"
            f"  Winner:{self.winner}"
        )
        self.board.print_board()


# class Gomoku:
#     def __init__(self):
#         self.board = None
#         self.captured_stones = []

#     def set_game(self, board_data: dict) -> None:
#         self.board = Board(board_data)

#     def print_board(self) -> None:
#         self.board.print_board()

#     def get_board(self) -> List[List[str]]:
#         return self.board.get_board()

#     def get_scores(self):
#         return [
#             {"player": self.last_player, "score": int(self.last_player_score / 2)},
#             {"player": self.next_player, "score": int(self.next_player_score / 2)},
#         ]

#     def reset_board(self) -> None:
#         self.board.reset_board()

#     def update_board(self, x: int, y: int, player: str) -> bool:
#         self.captured_stones = detect_captured_stones(self.board, x, y, player)
#         if self.captured_stones is not None:
#             self.remove_captured_stone(self.captured_stones)
#         elif self.is_doublethree(self.board, x, y, player):
#             return False
#         return True

#     def remove_captured_stone(self) -> None:
#         for captured in self.captured_stones:
#             self.board.set_value(captured["x"], captured["y"], EMPTY_SPACE)
#             # if captured["stone"] == self.next_player:
#             #     self.last_player_score += 1
#             # else:
#             #     self.next_player_score += 1
#             # self.record_history(
#             #     pair[0],
#             #     pair[1],
#             #     PLAYER_1 if player == PLAYER_2 else PLAYER_2,
#             #     "capture",
#             # )
#         print(
#             f"Capture score - {self.last_player}: {self.last_player_score}, {self.next_player}: {self.next_player_score}"
#         )

#     def is_doublethree(self, x: int, y: int, player: str) -> bool:
#         return detect_doublethree(self.board, x, y, player)

#     def get_legal_moves(self) -> list[tuple[int, int]]:
#         legal_moves = []
#         for y in range(NUM_LINES):
#             for x in range(NUM_LINES):
#                 if self.board.get_value(x, y) != EMPTY_SPACE:
#                     continue
#                 if self.board.enable_doublethree and detect_doublethree(
#                     self.board, x, y, self.board.next_player
#                 ):
#                     continue
#                 legal_moves.append((x, y))
#         return legal_moves

#     def place_stone(self, x: int, y: int, player: int) -> None:
#         self.board.set_value(x, y, player)

#     # def place_stone(self, x: int, y: int, player: str) -> bool:
#     #     if (
#     #         0 <= x < self.board_size
#     #         and 0 <= y < self.board_size
#     #         and self.board[x][y] == "."
#     #     ):
#     #         self.board.set_value(x, y, player)
#     #         # self.record_history(x, y, player, "place")
#     #         return True
#     #     return False

#     # def play_next_minmax(self):
#     #     """
#     #     Determine and play the next move for the AI using the Minimax algorithm.
#     #     """
#     #     start_time = time.time()

#     #     # Call the Minimax function to get the best move
#     #     score, col, row = minmax(
#     #         board=self.board,
#     #         depth=3,  # Adjust depth for performance vs. accuracy
#     #         is_maximizing=True,
#     #         alpha=float("-inf"),
#     #         beta=float("inf"),
#     #         player=self.board.next_player,
#     #         opponent=self.board.last_player,
#     #     )

#     #     # If a valid move is found, place the stone
#     #     if col != -1 and row != -1:
#     #         captured_stones = capture_opponent(
#     #             self.board, col, row, self.board.next_player
#     #         )
#     #         if captured_stones:
#     #             self.captured_stones = captured_stones
#     #             self.remove_captured_stone(captured_stones)
#     #         if self.place_stone(col, row, self.next_player):
#     #             print(f"AI placed stone at ({col}, {row}) with score {score}")

#     #     end_time = time.time()
#     #     elapsed_time_ms = (end_time - start_time) * 1000
#     #     print(f"AI chose move ({col}, {row}) in {elapsed_time_ms:.3f} ms")

#     #     # Return the move details
#     #     return {"coordinate": {"col": col, "row": row}, "stone": self.next_player}

#     #     # testing
#     #     print(self.next_player)
#     #     self.place_stone(1, 1, self.next_player)
#     #     return {"coordinate": {"x": 1, "y": 1}, "stone": self.next_player}

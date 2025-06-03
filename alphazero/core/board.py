from dataclasses import dataclass

import numpy as np
from core.game_config import (
    CAPTURE_GOAL,
    EMPTY_DOT,
    EMPTY_SPACE,
    PLAYER_1,
    PLAYER_2,
    PLAYER_X,
    get_pos,
    set_pos,
)
from core.rules.capture import detect_captured_stones
from core.rules.doublethree import detect_doublethree


@dataclass
class Board:
    pos: np.ndarray
    last_x: int | None = None
    last_y: int | None = None
    last_player: int = PLAYER_1
    next_player: int = PLAYER_2
    last_pts: int = 0
    next_pts: int = 0
    enable_capture: bool = True
    enable_doublethree: bool = True
    goal: int = CAPTURE_GOAL

    @classmethod
    def empty(
        cls,
        size: int = 19,
        enable_capture: bool = True,
        enable_doublethree: bool = True,
        goal: int = 10,
    ) -> "Board":
        return cls(
            pos=np.full((size, size), EMPTY_SPACE, np.uint8),
            goal=goal,
            enable_capture=enable_capture,
            enable_doublethree=enable_doublethree,
        )

    def is_legal_move(self, x: int, y: int, player: int) -> bool:
        if get_pos(self.pos, x, y) != EMPTY_SPACE:
            return False
        if self.enable_doublethree and detect_doublethree(self.pos, x, y, player):
            return False
        return True

    def apply_move(self, x: int, y: int, player: int) -> list[dict]:
        """합법이라고 가정"""
        set_pos(self.pos, x, y, player)
        self.last_x, self.last_y = x, y
        captures: list[dict] = []
        # 캡처
        if self.enable_capture:
            captures = detect_captured_stones(self, x, y, player)
            for s in captures:
                set_pos(self.pos, s["x"], s["y"], EMPTY_SPACE)
            if player == PLAYER_1:
                self.last_pts += len(captures) // 2
            else:
                self.next_pts += len(captures) // 2
        return captures

    def legal_moves(self, player: int) -> list[tuple[int, int]]:
        ys, xs = np.where(self.pos == EMPTY_SPACE)
        return [(x, y) for x, y in zip(xs, ys) if self.is_legal_move(x, y, player)]

    def update_from_dict(self, board_data: dict) -> None:
        """
        FastAPI 웹소켓이 보내주는 board_data(JSON)를
        현재 Board 인스턴스에 반영한다.
        """
        # 1) 점수・플래그
        self.goal = board_data["goal"]
        self.enable_capture = board_data["enableCapture"]
        self.enable_doublethree = board_data["enableDoubleThreeRestriction"]

        # 2) 플레이어 정보
        self.last_player = (
            PLAYER_1 if board_data["lastPlay"]["stone"] == PLAYER_X else PLAYER_2
        )
        self.next_player = (
            PLAYER_1 if board_data["nextPlayer"] == PLAYER_X else PLAYER_2
        )

        # 3) 점수
        self.last_pts = next(
            s["score"]
            for s in board_data["scores"]
            if s["player"] == board_data["lastPlay"]["stone"]
        )
        self.next_pts = next(
            s["score"]
            for s in board_data["scores"]
            if s["player"] == board_data["nextPlayer"]
        )

        # 4) 바둑판 배열 ('.' → EMPTY_SPACE 등)
        self.pos = np.array(
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

        # 5) 마지막 착수 좌표
        self.last_x = board_data["lastPlay"]["coordinate"].get("x")
        self.last_y = board_data["lastPlay"]["coordinate"].get("y")

    def print_board(self) -> None:
        """Prints the board with column letters (A-T) and row numbers (1-19)."""
        size = len(self.pos)
        column_labels = " ".join(chr(ord("A") + i) for i in range(size))
        print("   " + column_labels)
        for i, row in enumerate(self.pos):
            row_label = f"{i + 1:>2}"  # Right-align single-digit numbers
            row_str = " ".join(map(str, row))
            print(f"{row_label} {row_str}")


# class Board:
#     def __init__(
#         self,
#         board_data: dict,
#     ) -> None:
#         """Initialize the board from a provided game state dictionary."""
#         self.goal: int = board_data["goal"]
#         self.last_player: int = (
#             PLAYER_1 if board_data["lastPlay"]["stone"] == PLAYER_X else PLAYER_2
#         )
#         self.next_player: int = (
#             PLAYER_1 if board_data["nextPlayer"] == PLAYER_X else PLAYER_2
#         )
#         self._last_pts: int = next(
#             s["score"] for s in board_data["scores"] if s["player"] == PLAYER_X
#         )
#         self._next_pts = next(
#             s["score"] for s in board_data["scores"] if s["player"] == PLAYER_X
#         )

#         # Convert board from list of strings to NumPy array
#         self.position: np.array = np.array(
#             [
#                 [
#                     EMPTY_SPACE
#                     if cell == EMPTY_DOT
#                     else (PLAYER_1 if cell == PLAYER_X else PLAYER_2)
#                     for cell in row
#                 ]
#                 for row in board_data["board"]
#             ],
#             dtype=np.uint8,
#         )
#         self.enable_capture: bool = board_data["enableCapture"]
#         self.enable_doublethree: bool = board_data["enableDoubleThreeRestriction"]
#         self.last_x: int | None = None
#         self.last_y: int | None = None
#         last_x = board_data["lastPlay"]["coordinate"].get("x")
#         last_y = board_data["lastPlay"]["coordinate"].get("y")

#         if last_x is not None and last_y is not None:
#             self.last_x: int | None = last_x
#             self.last_y: int | None = last_y

#     def __getitem__(self, indices: tuple[int, int]) -> int:
#         """Get the value at a specific column and row."""
#         return self.position[indices]

#     def __setitem__(self, indices: tuple[int, int], value: int) -> None:
#         """Set the value at a specific column and row."""
#         self.position[indices] = value

#     def get_board(self) -> np.ndarray:
#         """Get the current board state."""
#         return self.position

#     def reset_board(self) -> None:
#         """Resets the board to an empty state."""
#         self.position.fill(EMPTY_SPACE)
#         self._last_pts = 0
#         self._next_pts = 0
#         self.last_x = None
#         self.last_y = None

#     def get_value(self, col: int, row: int) -> int:
#         """Get the value at a specific column and row."""
#         return self.position[row, col]

#     def switch_turn(self):
#         self.last_player, self.next_player = self.next_player, self.last_player
#         self._last_pts, self._next_pts = (
#             self._next_pts,
#             self._last_pts,
#         )

#     def set_value(self, col: int, row: int, value: int) -> None:
#         """Set the value at a specific column and row."""
#         self.position[row, col] = value
#         if value != EMPTY_SPACE:
#             self.update_last_move(col, row)
#             self.switch_turn()

#     def update_last_move(self, x: int, y: int) -> None:
#         self.last_x = x
#         self.last_y = y

#     def get_row(self, row: int) -> np.ndarray:
#         """Return a specific row."""
#         return self.position[row, :]

#     def get_column(self, col: int) -> np.ndarray:
#         """Return a specific column."""
#         return self.position[:, col]

#     def update_captured_stone(self, captured_stones: List[dict]) -> None:
#         """Removes captured stones from the board."""
#         for captured in captured_stones:
#             self.position[captured["y"], captured["x"]] = EMPTY_SPACE


#     # @staticmethod
#     # def print_board_param(board) -> None:
#     #     """Prints the board with column letters (A-T) and row numbers (1-19)."""
#     #     size = len(board)
#     #     column_labels = " ".join(chr(ord("A") + i) for i in range(size))
#     #     print("   " + column_labels)
#     #     for i, row in enumerate(board):
#     #         row_label = f"{i + 1:>2}"  # Right-align single-digit numbers
#     #         row_str = " ".join(map(str, row))
#     #         print(f"{row_label} {row_str}")


#     @property
#     def current_player(self) -> int:
#         return self.next_player

#     @property
#     def next_pts(self) -> int:
#         return self._next_pts

#     @property
#     def last_pts(self) -> int:
#         return self._last_pts

#     def get_legal_moves(self) -> List[tuple[int, int]]:
#         from core.rules.doublethree import detect_doublethree  # ← 함수 안 import

#         empty = self.position == EMPTY_SPACE
#         rows, cols = np.where(empty)
#         legal = []
#         for r, c in zip(rows, cols):
#             if self.enable_doublethree and detect_doublethree(
#                 self, c, r, self.current_player
#             ):
#                 continue
#             legal.append((r, c))
#         return legal

#     def print_legal_moves_grid(self) -> None:
#         """Prints the board showing legal moves as 1, others as 0."""
#         size = self.position.shape[0]
#         legal_positions = np.zeros((size, size), dtype=int)
#         for row, col in self.get_legal_moves():
#             legal_positions[row, col] = 1

#         column_labels = " ".join(chr(ord("A") + i) for i in range(size))
#         print("   " + column_labels)
#         for i, row in enumerate(legal_positions):
#             row_label = f"{i + 1:>2}"
#             row_str = " ".join(map(str, row))
#             print(f"{row_label} {row_str}")

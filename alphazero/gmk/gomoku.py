from __future__ import annotations

import copy
from dataclasses import dataclass
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


@dataclass(frozen=True)
class GameState:
    board: np.ndarray  # (19, 19) int8  ― 바둑판
    p1_pts: int  # 흑(PLAYER_1)의 포획 점수
    p2_pts: int  # 백(PLAYER_2)의 포획 점수
    next_player: int


class Gomoku:
    def __init__(self):
        self.row_count = NUM_LINES
        self.col_count = NUM_LINES
        self.last_captures = []
        self.action_size = self.row_count * self.col_count
        self.enable_doublethree = False
        self.enable_capture = False

    def get_initial_state(self) -> np.ndarray:
        board = np.zeros((self.row_count, self.col_count))
        return GameState(board=board, p1_pts=0, p2_pts=0, next_player=PLAYER_1)

    def get_next_state(
        self, state: GameState, action: Tuple[int, int], player: int
    ) -> GameState:
        board = copy.deepcopy(state.board)
        p1_pts, p2_pts = state.p1_pts, state.p2_pts

        x, y = action
        set_pos(board, x, y, player)

        if self.enable_capture:
            captures = detect_captured_stones(board, x, y, player)
            if len(captures) > 0:
                self.last_captures = captures
                for c in captures:
                    set_pos(board, c["x"], c["y"], EMPTY_SPACE)
                if player == PLAYER_1:
                    p1_pts += len(captures) // 2
                else:
                    p2_pts += len(captures) // 2

        return GameState(
            board=board,
            p1_pts=p1_pts,
            p2_pts=p2_pts,
            next_player=opponent_player(player),
        )

    def get_legal_moves(self, state: GameState) -> list[str]:
        """Returns a list of valid moves in [A1, B5, ...] format."""
        board = state.board
        moves = []
        # print("next_player", state.next_player)

        for y in range(self.row_count):
            for x in range(self.col_count):
                if get_pos(board, x, y) == EMPTY_SPACE:
                    if self.enable_doublethree and not detect_doublethree(
                        board, x, y, state.next_player
                    ):
                        # col = chr(ord("A") + x)
                        # row = str(y + 1)
                        # moves.append(f"{col}{row}")
                        moves.append(f"{chr(ord('A') + x)}{y + 1}")
                    elif not self.enable_doublethree:
                        moves.append(f"{chr(ord('A') + x)}{y + 1}")

        return moves

    def check_win(self, state: GameState, action: Tuple[int, int]) -> bool:
        if action is None:
            return False
        x, y = action
        player = get_pos(state.board, x, y)

        if check_local_gomoku(state.board, x, y, player):
            return True
        if player == PLAYER_1 and state.p1_pts >= CAPTURE_GOAL:
            return True
        if player == PLAYER_2 and state.p2_pts >= CAPTURE_GOAL:
            return True

        return False

    def get_value_and_terminated(self, state: GameState, action: Tuple[int, int]):
        if self.check_win(state, action):
            return 1, True
        # if not self.get_legal_moves(state, PLAYER_1) and not self.get_legal_moves(
        #     state, PLAYER_2
        # ):
        if not self.get_legal_moves(state):
            return 0, True
        return 0, False

    def print_board(self, state: GameState) -> None:
        """Prints the board with column letters (A-T) and row numbers (1-19)."""
        board = state.board
        column_labels = " ".join(chr(ord("A") + i) for i in range(self.col_count))
        print("   " + column_labels)
        for i, row in enumerate(board):
            print(f"{i + 1:>2} " + " ".join(str(int(c)) for c in row))
        print(f"Captures  P1:{state.p1_pts}  P2:{state.p2_pts}\n")


def convert_coordinates_to_index(coord: str) -> Tuple[int, int] | None:
    """
    Converts a coordinate string like 'A1' or 'T19' to (x, y) index Tuple.
    Returns None if the format is invalid.
    """
    if coord is None:
        return None
    if not (2 <= len(coord) and len(coord) <= 3):
        return None

    col_letter = coord[0]
    row_number = coord[1:]

    if not col_letter.isalpha() or not row_number.isdigit():
        return None

    x = ord(col_letter) - ord("A")
    y = int(row_number) - 1
    # print(x, y)

    if x < 0 or x >= NUM_LINES:
        return None

    if y < 0 or y >= NUM_LINES:
        return None

    return x, y


# gomoku = Gomoku()

# player = PLAYER_1

# state = gomoku.get_initial_state()

# # set_pos(state.board, 7, 7, PLAYER_1)
# # set_pos(state.board, 8, 7, PLAYER_2)
# # set_pos(state.board, 9, 7, PLAYER_2)

# # set_pos(state.board, 10, 4, PLAYER_1)
# # set_pos(state.board, 10, 5, PLAYER_2)
# # set_pos(state.board, 10, 6, PLAYER_2)

# # set_pos(state.board, 11, 7, PLAYER_2)
# # set_pos(state.board, 12, 7, PLAYER_2)
# # set_pos(state.board, 13, 7, PLAYER_1)

# # set_pos(state.board, 10, 8, PLAYER_2)
# # set_pos(state.board, 10, 9, PLAYER_2)
# # set_pos(state.board, 10, 10, PLAYER_1)


# while True:
#     gomoku.print_board(state)
#     valid_moves = gomoku.get_legal_moves(state)
#     print("valid_moves:", valid_moves)

#     action_str = input(f"{player} (e.g. A1): ").strip().upper()
#     if action_str not in valid_moves:
#         print("Action not valid -- invalid string")
#         continue

#     action_coord = convert_coordinates_to_index(action_str)
#     if action_coord is None:
#         print("Action not valid -- 2 ")
#         continue

#     state = gomoku.get_next_state(state, action_coord, player)

#     # gomoku.print_board(state)
#     value, is_terminal = gomoku.get_value_and_terminated(state, action_coord)

#     if is_terminal:
#         print(state)
#         if value == 1:
#             print(player, "won")
#         else:
#             print("draw")
#         break

#     player = opponent_player(player)

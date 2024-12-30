from config import *
from src.game.game_util import make_list_to_direction
from src.interface.model.capture import dfs_capture, remove_captured_list
from src.interface.model.check_doublethree import (
    check_continous_from_position,
    check_next_only_range,
)
from src.game.board import Board, print_colored_text
from src.game.Player import Player


class GameModel:
    def __init__(self):
        self.board = Board()
        self.player1 = Player(PLAYER_1)
        self.player2 = Player(PLAYER_2)
        self.record_count = 1
        # TODO: integrate with Board
        self.record = []
        # TODO: integrate with Board
        self.trace = []
        self.game_data = []

    # TODO: keep this, or move this into initializer if needed
    def set_config(self, options):
        self.options = options
        print(options)

    def is_draw(self) -> bool:
        # loop over board square
        for row in range(NUM_LINES):
            for col in range(NUM_LINES):
                if self.board.is_empty_square(col, row):
                    return False
        return True

    def make_move(self, col: int, row: int) -> object:
        # create new board instance that inherits from the current state
        board = Board(self.board)

        # make move
        board.position[row][col] = self.board.turn

        return board, (col, row)

    def place_stone(self, x, y, captured_list=None):
        print(f"x: {x}, y: {y}")
        self.board, action = self.make_move(x, y)
        self.record_trace(x, y)
        # self.trace.append(self.game_logic.board)
        if captured_list is not None:
            remove_captured_list(self.board, captured_list)
        # self.change_player_turn()
        self.game_data.append((self.board, action))

    def record_trace(self, x, y):
        self.record.append(((x, y), self.board.turn, self.record_count))
        self.record_count += 1
        self.trace.append(self.board)

    def undo_last_move(self):
        if self.trace:  # Checks if the trace list is not empty
            self.record_count -= 1
            self.trace.pop()
            self.record.pop()
            if self.trace:
                self.board = self.trace[-1]
            else:
                self.board.position = [["."] * NUM_LINES for _ in range(NUM_LINES)]
            return True
        else:
            return False

    def change_player_turn(self):
        self.board.turn = PLAYER_2 if self.board.turn == PLAYER_1 else PLAYER_1

    def check_doublethree(self, x, y):
        def find_continuous_range():
            for dir in DIRECTIONS:
                if check_continous_from_position(
                    self.board, x, y, dir, self.board.turn
                ):
                    return (
                        make_list_to_direction(
                            self.board, x, y, dir, 5, self.board.turn
                        ),
                        dir,
                    )
            return None, None

        continous_range, direction = find_continuous_range()
        if direction is None:
            for i in range(len(DIRECTIONS) // 2):
                continous_range = check_next_only_range(
                    self.board, x, y, DIRECTIONS[i], self.board.turn
                )
                if continous_range:
                    direction = DIRECTIONS[i]
                    break
        if direction is not None:
            directions_copy = [
                d
                for d in DIRECTIONS
                if d != direction and d != (-direction[0], -direction[1])
            ]
            for one_place in continous_range:
                for dir in directions_copy:
                    if check_continous_from_position(
                        self.board, one_place[0], one_place[1], dir, self.board.turn
                    ):
                        return True
                    elif check_next_only_range(
                        self.board, one_place[0], one_place[1], dir, self.board.turn
                    ):
                        return True
        return False

    def capture_opponent(self, x, y):
        captured_list = []
        for dir in DIRECTIONS:
            # print(dir)
            if dfs_capture(self.board, x, y, self.board.turn, dir, 1) == True:
                captured_list = [
                    (x + dir[0], y + dir[1]),
                    (x + (dir[0] * 2), y + (dir[1] * 2)),
                ]
        return captured_list

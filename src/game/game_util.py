from config import NUM_LINES
from src.game.board import Board


def is_valid_position(position):
    # Check if the position (x, y) is within the bounds of the board.
    return 0 <= position[0] < NUM_LINES and 0 <= position[1] < NUM_LINES


def make_list_to_direction(board: Board, x, y, dir, n, player):
    return_list = []
    return_list.append((x, y))
    for i in range(1, n):
        new_x, new_y = x + dir[0] * i, y + dir[1] * i
        if (
            is_valid_position((new_x, new_y))
            and board.get_value(new_x, new_y) == player
        ):
            return_list.append((new_x, new_y))

    return return_list

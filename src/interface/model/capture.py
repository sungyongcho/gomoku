from src.game.board import Board
from config import *
from src.game.game_util import *


def dfs_capture(board: Board, x, y, player, direction, count):
    nx = x + direction[0]
    if nx > NUM_LINES:
        return False
    ny = y + direction[1]
    if ny > NUM_LINES:
        return False

    if count == 3:
        if is_valid_position((nx, ny)) == True and board.get_value(nx, ny) == player:
            return True
        else:
            return False

    if not is_valid_position((nx, ny)):
        return False
    if board.get_value(nx, ny) != (PLAYER_2 if player == PLAYER_1 else PLAYER_1):
        return False

    return dfs_capture(board, nx, ny, player, direction, count + 1)


def remove_captured_list(board: Board, captured_list):
    for pair in captured_list:
        board.set_value(pair[0], pair[1], EMPTY_SQUARE)

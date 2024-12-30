from config import *
from src.game.board import Board

from src.game.game_util import *


def dfs(board: Board, x, y, player, direction, count, player_count):
    nx, ny = x + direction[0], y + direction[1]
    if not is_valid_position((nx, ny)) or count > 4:
        return False

    if 3 <= count and count <= 4:
        if (
            player_count >= 3
            and is_valid_position((nx, ny)) == True
            and board.get_value(nx, ny)
            != (PLAYER_2 if player == PLAYER_1 else PLAYER_1)
        ):
            return True

    if not is_valid_position((nx, ny)):
        return False

    if board.get_value(nx, ny) == (PLAYER_2 if player == PLAYER_1 else PLAYER_1):
        return False

    if board.get_value(nx, ny) == player:
        player_count += 1

    return dfs(board, nx, ny, player, direction, count + 1, player_count)


def check_next_only_range(board: Board, x, y, dir, player):
    return_list = []
    print("direction", dir, (-dir[0], -dir[1]), player)
    if (
        is_valid_position((x + dir[0], y + dir[1])) is False
        or is_valid_position((x + (dir[0] * 2), y + (dir[1] * 2))) is False
        or is_valid_position((x - dir[0], y - dir[1])) is False
        or is_valid_position((x - (dir[0] * 2), y - (dir[1] * 2))) is False
    ):
        return None

    def check_one():
        nonlocal return_list
        if (
            board.get_value(x + dir[0], y + dir[1])
            == board.get_value(x - dir[0], y - dir[1])
            == player
        ):
            if board.get_value(x + (dir[0] * 2), y + (dir[1] * 2)) != (
                PLAYER_2 if player == PLAYER_1 else PLAYER_1
            ) and board.get_value(x - (dir[0] * 2), y - (dir[1] * 2)) != (
                PLAYER_2 if player == PLAYER_1 else PLAYER_1
            ):
                return_list = make_list_to_direction(
                    board, x, y, (-dir[0], -dir[1]), 3, player
                )
                return_list += make_list_to_direction(board, x, y, dir, 3, player)

        return return_list

    def check_dfs(dir):
        nonlocal return_list
        if (
            dfs(board, x, y, player, dir, 2, 2) == True
            and (
                board.get_value(x - dir[0], y - dir[1])
                == board.get_value(x, y)
                == player
            )
            and (
                board.get_value(x - (dir[0] * 2), y - (dir[1] * 2))
                != (PLAYER_2 if player == PLAYER_1 else PLAYER_1)
            )
        ):
            return_list = make_list_to_direction(board, x, y, dir, 3, player)
            return_list += make_list_to_direction(
                board, x, y, (-dir[0], -dir[1]), 3, player
            )

        return return_list

    return_list = check_one() or check_dfs(dir) or check_dfs((-dir[0], -dir[1]))

    return list(set(return_list))


def check_continous_from_position(board: Board, x, y, dir, player):
    if (
        dfs(board, x, y, player, dir, 1, 1)
        and is_valid_position((x - dir[0], y - dir[1]))
        and board.get_value(x - dir[0], y - dir[1]) == EMPTY_SQUARE
    ):
        return True
    return False

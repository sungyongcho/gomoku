from config import *


def dfs_capture(board, x, y, player, direction, count) -> bool:
    nx = x + direction[0]
    ny = y + direction[1]

    if count == 3:
        if board.get_value(nx, ny) == player:
            return True
        else:
            return False

    if board.get_value(nx, ny) != (PLAYER_2 if player == PLAYER_1 else PLAYER_1):
        return False

    return dfs_capture(board, nx, ny, player, direction, count + 1)


def capture_opponent(board, x, y, player):
    print("capture_opponent", (x, y), player)
    captured_stones = []
    for dir in DIRECTIONS:
        if x + dir[0] * 3 >= NUM_LINES or x + dir[0] * 3 < 0:
            continue
        if y + dir[1] * 3 >= NUM_LINES or y + dir[1] * 3 < 0:
            continue
        print(dir, x + dir[0] * 3, y + dir[1] * 3)
        if dfs_capture(board, x, y, player, dir, 1):
            captured_stones.append((x + dir[0], y + dir[1]))
            captured_stones.append((x + (dir[0] * 2), y + (dir[1] * 2)))
    print(captured_stones)
    return captured_stones

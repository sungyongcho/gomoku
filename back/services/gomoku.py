import time

BOARD_SIZE = 19
board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
history = []


def convert_board_for_print():
    """Converts the board to a human-readable string."""
    symbols = {0: ".", "X": "X", "O": "O"}  # Mapping of values to symbols
    board_to_print = ""
    for row in board:
        board_to_print += "".join(symbols[cell] for cell in row) + "\n"
    return board_to_print


def get_board():
    return convert_board_for_print()


def update_board(x: int, y: int, player: str):
    check_capture(x, y, player)
    place_stone(x, y, player)


def place_stone(x: int, y: int, player: str) -> bool:
    """Updates the board with a new move and measures processing time."""
    if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[y][x] == 0:
        board[y][x] = player
        record_history(x, y, player)
        return True

    return False


def check_capture(x: int, y: int, player: str):
    pass


def record_history(x: int, y: int, player: str) -> None:
    history.append({"x": x, "y": y, "player": player})


def print_history():
    print(history)


def play_next():
    start_time = time.time()  # Record the start time

    last_move = history[len(history) - 1]
    x, y = last_move["x"], last_move["y"]
    place_stone(x + 1, y + 1, "O")
    end_time = time.time()  # Record the end time
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to ms
    print(f"Update processed in {elapsed_time_ms:.3f} ms")
    pass


def reset_board():
    """Resets the board to an empty state."""
    global board
    board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

BOARD_SIZE = 19
board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def convert_board_for_print():
    """Converts the board to a human-readable string."""
    symbols = {0: ".", "X": "X", "O": "O"}  # Mapping of values to symbols
    board_to_print = ""
    for row in board:
        board_to_print += "".join(symbols[cell] for cell in row) + "\n"
    return board_to_print


def get_board():
    return convert_board_for_print()


def update_board(x: int, y: int, player: str) -> bool:
    """Updates the board with a new move."""
    if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[y][x] == 0:
        board[y][x] = player
        return True
    return False


def reset_board():
    """Resets the board to an empty state."""
    global board
    board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

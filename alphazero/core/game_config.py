# board
NUM_LINES = 19
EMPTY_DOT = "."

PLAYER_X = "X"  # Black stone
PLAYER_O = "O"  # White stone
EMPTY_SPACE = 0
PLAYER_1 = 1  # Black stone
PLAYER_2 = 2  # White stone
OUT_OF_BOUNDS = 3
CAPTURE_GOAL = 5


def opponent_player(player):
    return PLAYER_2 if player == PLAYER_1 else PLAYER_1


# direction for doublethree
NORTH = (0, -1)
NORTHEAST = (1, -1)
EAST = (1, 0)
SOUTHEAST = (1, 1)
SOUTH = (0, 1)
SOUTHWEST = (-1, 1)
WEST = (-1, 0)
NORTHWEST = (-1, -1)

DIRECTIONS = [NORTH, NORTHEAST, EAST, SOUTHEAST, SOUTH, SOUTHWEST, WEST, NORTHWEST]

UNIQUE_DIRECTIONS = [
    NORTH,
    NORTHEAST,
    EAST,
    SOUTHEAST,
]

# Constants
import math

import pygame

RESET = 1

BLACK = (0, 0, 0)
BLACK_TRANSPARENT = pygame.Color(0, 0, 0, 128)

WHITE = (255, 255, 255)
WHITE_TRANSPARENT = pygame.Color(255, 255, 255, 128)

SCREEN_WIDTH, SCREEN_HEIGHT = 1600, 1000
NUM_LINES = 9
LINE_COLOR = BLACK
BACKGROUND_COLOR = WHITE
PLAYER_1 = "X"  # Black stone
PLAYER_2 = "O"  # White stone
EMPTY_SQUARE = "."

GRID_START_X = (5 * SCREEN_WIDTH / 8) / 10
GRID_START_Y = SCREEN_HEIGHT / 10

# GRID_END_X = GRID_START_X + SCREEN_WIDTH // 2
# GRID_END_Y = GRID_START_Y + SCREEN_HEIGHT // 1.25

CELL_SIZE_X = (SCREEN_WIDTH // 2) / NUM_LINES
CELL_SIZE_Y = (SCREEN_HEIGHT // 1.25) / NUM_LINES


right_pane_begin_x = 5 * SCREEN_WIDTH / 8
right_pane_begin_y = 0
right_pane_width = 3 * SCREEN_WIDTH / 8
right_pane_height = SCREEN_HEIGHT

time_width = right_pane_width / 3
time_height = right_pane_height / 10

scorebox_width = 4 * right_pane_width / 5
scorebox_height = right_pane_height / 3

log_width = 5 * right_pane_width / 6
log_height = 4 * right_pane_height / 10


# doublethree direction
NORTH = (0, -1)
NORTHEAST = (1, -1)
EAST = (1, 0)
SOUTHEAST = (1, 1)
SOUTH = (0, 1)
SOUTHWEST = (-1, 1)
WEST = (-1, 0)
NORTHWEST = (-1, -1)

DIRECTIONS = [NORTH, NORTHEAST, EAST, SOUTHEAST, SOUTH, SOUTHWEST, WEST, NORTHWEST]


# MENU

MAIN_MENU = 1
OPTIONS_MENU = 2

# [Neural Network]
INPUT_SHAPE = NUM_LINES * NUM_LINES
OUTPUT_SHAPE = NUM_LINES
NUM_EPISODES = 1000
BATCH_SIZE = 5

# [MCTS]

# [Path]

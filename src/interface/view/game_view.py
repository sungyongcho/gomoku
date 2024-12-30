import pygame
import pygame_gui
from config import *
from pygame_gui.elements.ui_text_box import UITextBox
from src.game.board import Board
from src.interface.view.game_menu import GameMenu
from src.interface.view.modal_window import ModalWindow


class GameView:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.bg = pygame.Surface((self.width, self.height))
        self.font_name = pygame.font.match_font("arial")
        self.reset_requested = False
        self._initialize_game_view()
        self._initialize_size()
        self.player_turn = PLAYER_1

    def update_board_and_player_turn(self, board: Board, record):
        self.board = board
        self.record = record

    def _initialize_size(self):
        self.grid_start_x = GRID_START_X
        self.grid_start_y = GRID_START_Y
        self.grid_width = SCREEN_WIDTH // 2
        self.grid_height = SCREEN_HEIGHT // 1.25

    def _initialize_game_view(self):
        # for pygame_gui (log textbox)
        self.ui_manager = pygame_gui.UIManager(
            (self.width, self.height), "resources/log_theme.json"
        )
        self.ui_manager.get_theme().load_theme("resources/modal_window_theme.json")

        self.modal_window = ModalWindow(self.ui_manager, (self.width, self.height))

        self._initialize_gameboard()
        self._initialize_right_pane()

    def _initialize_gameboard(self):
        self.board_surface = pygame.Surface(
            (5 * SCREEN_WIDTH / 8, SCREEN_HEIGHT), pygame.SRCALPHA
        )

    def _initialize_right_pane(self):
        self.right_pane = pygame.Surface(
            (right_pane_width, right_pane_height), pygame.SRCALPHA
        )
        self.right_pane_rect = self.right_pane.get_rect(
            topleft=(right_pane_begin_x, right_pane_begin_y)
        )
        self._initialize_time_rect()
        self._initialize_scorebox_rect()
        self._initialize_log_rect()
        self._initialize_text_box()

    def _initialize_time_rect(self):
        self.time_rect = pygame.Rect(
            self.right_pane_rect.centerx - scorebox_width / 2,
            time_height / 2,
            time_width,
            time_height,
        )

        self.goal_rect = pygame.Rect(
            self.right_pane_rect.centerx
            + (scorebox_width - self.time_rect.width * 2) / 2,
            time_height / 2,
            time_width,
            time_height,
        )

    def _initialize_scorebox_rect(self):
        self.scorebox_rect = pygame.Rect(
            self.right_pane_rect.centerx - scorebox_width / 2,
            self.time_rect.bottom,
            scorebox_width,
            scorebox_height,
        )
        self.p1_name_rect = pygame.Rect(
            self.scorebox_rect.left,
            self.scorebox_rect.top,
            self.scorebox_rect.width / 2,
            self.scorebox_rect.height / 2 * 0.8,
        )
        self.p2_name_rect = pygame.Rect(
            self.scorebox_rect.centerx,
            self.scorebox_rect.top,
            self.scorebox_rect.width / 2,
            self.scorebox_rect.height / 2 * 0.8,
        )

        self.cursor_left = pygame.Rect(
            self.scorebox_rect.left,
            self.scorebox_rect.top + self.scorebox_rect.height / 2 * 0.8,
            self.scorebox_rect.width / 2,
            self.scorebox_rect.height / 2 * 0.2,
        )

        self.cursor_right = pygame.Rect(
            self.scorebox_rect.centerx,
            self.scorebox_rect.top + self.scorebox_rect.height / 2 * 0.8,
            self.scorebox_rect.width / 2,
            self.scorebox_rect.height / 2 * 0.2,
        )

        self.p1_score_rect = pygame.Rect(
            self.scorebox_rect.left,
            self.scorebox_rect.top + self.scorebox_rect.height / 2,
            self.scorebox_rect.width / 2,
            self.scorebox_rect.height / 2 * 0.6,
        )
        self.p2_score_rect = pygame.Rect(
            self.scorebox_rect.centerx,
            self.scorebox_rect.top + self.scorebox_rect.height / 2,
            self.scorebox_rect.width / 2,
            self.scorebox_rect.height / 2 * 0.6,
        )

        self.pause_rect = pygame.Rect(
            self.scorebox_rect.centerx - self.scorebox_rect.width / 4,
            self.p1_score_rect.bottom,
            self.scorebox_rect.width / 2,
            self.scorebox_rect.height / 2 * 0.5,
        )

    def _initialize_log_rect(self):
        self.log_rect = pygame.Rect(
            self.right_pane_rect.centerx - log_width / 2,
            self.scorebox_rect.bottom + self.right_pane_rect.height / 20,
            log_width,
            log_height,
        )

    def _initialize_text_box(self):
        self.text_box = UITextBox(
            html_text="<body><font color=#E0E080></font>",
            relative_rect=self.log_rect,
            manager=self.ui_manager,
        )

    def draw_text(self, text, size, color, x, y):
        font = pygame.font.Font(self.font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        self.screen.blit(text_surface, text_rect)

    def find_index_record(self, x, y):
        for i, item in enumerate(self.record):
            if item[0] == (x, y) and not self.board.is_empty_square(x, y):
                return item[2]
        return -1

    def draw_stone(self, x, y, target, color, thickness=0):
        initial_size = 6  # Adjust as needed
        max_lines = 20  # Adjust as needed
        size_increase = 1  # Adjust as needed

        circle_size = initial_size + (max_lines - NUM_LINES) * size_increase

        if color == BLACK or color == BLACK_TRANSPARENT:
            pygame.draw.circle(
                target,
                color,
                (
                    x * CELL_SIZE_X + GRID_START_X + CELL_SIZE_X // 2,
                    y * CELL_SIZE_Y + GRID_START_Y + CELL_SIZE_Y // 2,
                ),
                circle_size,
            )
        elif color == WHITE or color == WHITE_TRANSPARENT:
            pygame.draw.circle(
                target,
                color,
                (
                    x * CELL_SIZE_X + GRID_START_X + CELL_SIZE_X // 2,
                    y * CELL_SIZE_Y + GRID_START_Y + CELL_SIZE_Y // 2,
                ),
                circle_size,
            )
            pygame.draw.circle(
                target,
                BLACK,
                (
                    x * CELL_SIZE_X + GRID_START_X + CELL_SIZE_X // 2,
                    y * CELL_SIZE_Y + GRID_START_Y + CELL_SIZE_Y // 2,
                ),
                circle_size,
                thickness,
            )

        record_index = self.find_index_record(x, y)
        if record_index != -1:
            # Create a font object
            font = pygame.font.Font(None, 36)  # You can adjust the font size as needed

            # Render the text
            record_color = (255, 255, 255) if color == BLACK else (0, 0, 0)
            text_surface = font.render(
                str(record_index),
                True,
                (255, 0, 0) if record_index == len(self.record) else record_color,
            )  # White color

            # Get the rect of the text surface
            text_rect = text_surface.get_rect(
                center=(
                    x * CELL_SIZE_X + GRID_START_X + CELL_SIZE_X // 2,
                    y * CELL_SIZE_Y + GRID_START_Y + CELL_SIZE_Y // 2,
                )
            )

            # Draw the text surface onto the target surface
            target.blit(text_surface, text_rect)

    def new(self):
        self.__init__(self.width, self.height)
        game_menu = GameMenu(self.screen, self.width, self.height)
        return game_menu.wait_for_key()

    def _convert_mouse_to_grid(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_x -= GRID_START_X
        mouse_y -= GRID_START_Y
        grid_x = 0
        grid_y = 0
        if (
            0 <= mouse_x < CELL_SIZE_X * NUM_LINES
            and 0 <= mouse_y < CELL_SIZE_Y * NUM_LINES
            and mouse_x <= SCREEN_WIDTH - GRID_START_X
            and mouse_y <= SCREEN_HEIGHT - GRID_START_Y
        ):
            grid_x = mouse_x // CELL_SIZE_X
            grid_y = mouse_y // CELL_SIZE_Y

        return int(grid_x), int(grid_y)

    def _anchor_mouse_stones(self):
        self.board_surface.fill(BACKGROUND_COLOR)
        grid_x, grid_y = self._convert_mouse_to_grid()

        # for anchor
        if self.board.turn == PLAYER_1:
            self.draw_stone(grid_x, grid_y, self.board_surface, BLACK)
        elif self.board.turn == PLAYER_2:
            self.draw_stone(grid_x, grid_y, self.board_surface, WHITE, 1)

    def _draw_placed_stones(self):
        # for drawing already placed dots
        for x in range(NUM_LINES):
            for y in range(NUM_LINES):
                if self.board.get_value(x, y) == "X":
                    self.draw_stone(x, y, self.screen, BLACK)
                elif self.board.get_value(x, y) == "O":
                    self.draw_stone(x, y, self.screen, WHITE, 1)

    def create_grid(self):
        cell_width = self.grid_width / NUM_LINES
        cell_height = self.grid_height / NUM_LINES

        font_size = min(self.grid_width // NUM_LINES, SCREEN_HEIGHT // 35)
        font = pygame.font.Font(None, font_size)

        for i in range(1, NUM_LINES + 1):
            # Vertical lines
            pygame.draw.line(
                self.screen,
                LINE_COLOR,
                (
                    self.grid_start_x + i * cell_width - (cell_width / 2),
                    self.grid_start_y,
                ),
                (
                    self.grid_start_x + i * cell_width - (cell_width / 2),
                    self.grid_start_y + self.grid_height,
                ),
            )
            text = font.render(chr(64 + i), True, (0, 0, 0))

            # Get the width and height of the text surface
            text_width, text_height = text.get_size()

            # Calculate the x and y coordinates to center the text
            x = (
                self.grid_start_x
                + (i - 1) * cell_width
                + cell_width / 2
                - text_width / 2
            )
            y = self.grid_start_y - text_height * 1.2

            # Draw the text on the screen
            self.screen.blit(text, (x, y))

        for i in range(1, NUM_LINES + 1):
            # Horizontal lines
            pygame.draw.line(
                self.screen,
                LINE_COLOR,
                (
                    self.grid_start_x,
                    self.grid_start_y + i * cell_height - (cell_height / 2),
                ),
                (
                    self.grid_start_x + self.grid_width,
                    self.grid_start_y + i * cell_height - (cell_height / 2),
                ),
            )
            # Create a text surface object for horizontal lines
            text = font.render(str(i), True, (0, 0, 0))

            # Get the width and height of the text surface
            text_width, text_height = text.get_size()

            # Calculate the x and y coordinates to center the text
            x = self.grid_start_x - text_width * 1.2
            y = (
                self.grid_start_y
                + (i - 1) * cell_height
                + cell_height / 2
                - text_height / 2
            )

            # Draw the text on the screen
            self.screen.blit(text, (x, y))

    def display_time(self):
        pygame.draw.rect(self.screen, BACKGROUND_COLOR, self.time_rect)
        pygame.draw.rect(self.screen, BACKGROUND_COLOR, self.goal_rect)
        # Get the elapsed time since pygame started

        elapsed_time_millis = pygame.time.get_ticks()
        # Convert the time to minutes and seconds

        if not self.modal_window.is_open:
            elapsed_time = divmod(elapsed_time_millis // 1000, 60)
            # Format the time as "00:00"

            self.formatted_time = "{:02}:{:02}".format(*elapsed_time)
            # Draw the time
        self.draw_text(
            str(self.formatted_time),
            6 * (self.width // 200),  # for responsive
            BLACK,
            self.time_rect.centerx,
            self.time_rect.centery // 1.5,
        )

        self.draw_text(
            str("goal: 10"),
            6 * (self.width // 200),  # for responsive
            BLACK,
            self.goal_rect.centerx,
            self.goal_rect.centery // 1.5,
        )

    def display_score(self):
        self.draw_text(
            f"{self.game_model.player1.captured}",
            10 * (self.width // 200),
            BLACK,
            self.p1_score_rect.centerx,
            self.p1_score_rect.top,
        )
        self.draw_text(
            f"{self.game_model.player2.captured}",
            10 * (self.width // 200),
            BLACK,
            self.p2_score_rect.centerx,
            self.p2_score_rect.top,
        )

        self.draw_text(
            "PAUSE",
            8 * (self.width // 200),
            BLACK,
            self.pause_rect.centerx,
            self.pause_rect.top,
        )

    def display_log(self):
        pygame.draw.rect(self.screen, BACKGROUND_COLOR, self.log_rect)

    def display_scorebox(self):
        pygame.draw.rect(self.screen, BLACK, self.p1_name_rect, 2)
        pygame.draw.rect(self.screen, BLACK, self.p2_name_rect, 2)
        pygame.draw.rect(
            self.screen,
            BACKGROUND_COLOR,
            self.cursor_left,
        )
        pygame.draw.rect(
            self.screen,
            BACKGROUND_COLOR,
            self.cursor_right,
        )
        pygame.draw.rect(self.screen, BLACK, self.p1_score_rect, 2)
        pygame.draw.rect(self.screen, BLACK, self.p2_score_rect, 2)
        pygame.draw.rect(self.screen, BLACK, self.pause_rect, 2)
        # self.display_score()

    def _display_right_pane(self):
        self.right_pane.fill((128, 128, 128, 128))
        self.display_time()
        self.display_scorebox()
        self.display_log()

    def draw(self, mode):
        self.screen.blit(self.board_surface, (0, 0))
        self.screen.blit(self.right_pane, (right_pane_begin_x, right_pane_begin_y))
        if not self.modal_window.is_open and not mode == "selfplay":
            self._anchor_mouse_stones()
        self.create_grid()
        self._draw_placed_stones()
        self._display_right_pane()
        self.ui_manager.update(0.01)
        self.ui_manager.draw_ui(window_surface=self.screen)
        pygame.display.update()

    def append_log(self, string):
        html_string = string + "<br>"
        self.text_box.append_html_text(html_string)

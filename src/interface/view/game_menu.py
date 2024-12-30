import pygame
import pygame_gui
from config import *


class GameMenu:
    def __init__(self, screen, screen_width, screen_height):
        self.screen = screen
        self.width = screen_width
        self.height = screen_height
        self.manager = pygame_gui.UIManager(
            (SCREEN_WIDTH, SCREEN_HEIGHT), "resources/log_theme.json"
        )
        self.init_main_menu()
        self.init_options_menu()
        self.init_texts()
        self.menu = MAIN_MENU

    def init_main_menu(self):
        button_width = self.width / 6
        button_height = self.height / 12
        button_padding_horiz = button_height / 3
        # Create the main menu button
        self.button_to_singleplay = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                ((self.width - button_width) // 2, self.height / 2),
                (button_width, button_height),
            ),
            text="Single Player (vs AI)",
            manager=self.manager,
        )

        self.button_to_multiplay = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (
                    (self.width - button_width) // 2,
                    self.height / 2 + (button_height + button_padding_horiz),
                ),
                (button_width, button_height),
            ),
            text="Multiplayer",
            manager=self.manager,
        )

        self.button_to_options = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (
                    (self.width - button_width) // 2,
                    self.height / 2 + (2 * (button_height + button_padding_horiz)),
                ),
                (button_width, button_height),
            ),
            text="Options",
            manager=self.manager,
        )

    def init_options_menu(self):
        # Slider and label for capture stone

        capture_stone_text_rect = pygame.Rect(
            int(self.width / 5), int(self.height * 0.3), 180, 30
        )

        self.capture_stone_text = pygame_gui.elements.UILabel(
            relative_rect=capture_stone_text_rect,
            text="Capture Stone Number",
            manager=self.manager,
        )

        capture_stone_slider_rect = pygame.Rect(
            capture_stone_text_rect.right,
            int(self.height * 0.3),
            self.width / 2,
            30,
        )

        self.capture_stone_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=capture_stone_slider_rect,
            start_value=10.0,
            value_range=(10.0, 50.0),
            manager=self.manager,
            object_id="#capture_stone_slider",
            click_increment=2,
        )
        self.capture_stone_label = pygame_gui.elements.UILabel(
            pygame.Rect(
                (
                    capture_stone_slider_rect.topleft[0],
                    capture_stone_slider_rect.topleft[1] - 30,
                ),
                (40, 40),
            ),
            str(int(self.capture_stone_slider.get_current_value())),
            self.manager,
        )

        advantage_text_rect = pygame.Rect(
            int(self.width / 5), int(self.height * 0.4), 180, 30
        )

        self.advantage_text = pygame_gui.elements.UILabel(
            relative_rect=advantage_text_rect,
            text="Advantage",
            manager=self.manager,
        )
        # Slider and label for black capture
        black_slider_rect = pygame.Rect(
            advantage_text_rect.right, int(self.height * 0.4), int(self.width / 4.5), 30
        )
        self.black_capture_slider = pygame_gui.elements.UIHorizontalSlider(
            black_slider_rect,
            0,
            (0, 6),
            self.manager,
            object_id="#black_capture_slider",
            click_increment=2,
        )
        self.black_capture_label = pygame_gui.elements.UILabel(
            pygame.Rect(
                (black_slider_rect.topleft[0], black_slider_rect.topleft[1] - 30),
                (40, 40),
            ),
            str(int(self.black_capture_slider.get_current_value())),
            self.manager,
        )

        # Slider and label for black capture
        white_slider_rect = pygame.Rect(
            black_slider_rect.right + self.width / 16,
            int(self.height * 0.4),
            int(self.width / 4.5),
            30,
        )
        self.white_capture_slider = pygame_gui.elements.UIHorizontalSlider(
            white_slider_rect,
            0,
            (0, 6),
            self.manager,
            object_id="#white_capture_slider",
            click_increment=2,
        )
        self.white_capture_label = pygame_gui.elements.UILabel(
            pygame.Rect(
                (white_slider_rect.topleft[0], white_slider_rect.topleft[1] - 30),
                (40, 40),
            ),
            str(int(self.white_capture_slider.get_current_value())),
            self.manager,
        )

        capture_enable_text_rect = pygame.Rect(
            int(self.width / 5), int(self.height * 0.5), 180, 30
        )

        self.capture_enable_text = pygame_gui.elements.UILabel(
            relative_rect=capture_enable_text_rect,
            text="Capture",
            manager=self.manager,
        )

        capture_buttion_rect = pygame.Rect(
            (capture_enable_text_rect.right, int(self.height * 0.5)), (200, 50)
        )
        # "Capture" button
        self.capture_button = pygame_gui.elements.UIButton(
            capture_buttion_rect,
            "Enable",
            self.manager,
            object_id="#capture_button",
        )

        doublethree_text_rect = pygame.Rect(
            capture_buttion_rect.right, int(self.height * 0.5), 180, 30
        )

        self.doublethree_text_label = pygame_gui.elements.UILabel(
            relative_rect=doublethree_text_rect,
            text="Double Three",
            manager=self.manager,
        )

        doublethree_button_rect = pygame.Rect(
            (doublethree_text_rect.right, int(self.height * 0.5)), (200, 50)
        )
        # "Double Three" button
        self.doublethree_button = pygame_gui.elements.UIButton(
            doublethree_button_rect,
            "Enable",
            self.manager,
            object_id="#doublethree_button",
        )

        dropdown_label_rect = pygame.Rect(
            int(self.width / 5), int(self.height * 0.6), 180, 30
        )

        self.dropdown_label = pygame_gui.elements.UILabel(
            relative_rect=dropdown_label_rect,
            text="Mode",
            manager=self.manager,
        )

        # Dropdown menu
        options = [
            "Standard",
            "Pro",
            "Swap",
            "Swap2",
        ]  # Replace with your desired options
        dropdown_rect = pygame.Rect((self.width / 2, int(self.height * 0.6)), (200, 50))
        self.dropdown_menu = pygame_gui.elements.UIDropDownMenu(
            options,
            options[0],
            dropdown_rect,
            self.manager,
            object_id="#dropdown_menu",
        )

        # "Back" button at the bottom
        self.button_to_main = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (int(self.width / 2) - 100, int(self.height * 0.8)), (200, 50)
            ),
            text="Back",
            manager=self.manager,
        )

    def init_texts(self):
        # Create UI Label elements for texts on different screens
        text_main_rect = (400, 100)
        self.text_main = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (self.width / 2 - text_main_rect[0] / 2, int(self.height * 0.15)),
                text_main_rect,
            ),
            text="오목 // Gomoku",
            manager=self.manager,
            object_id="#main_title",
        )

        self.text_options = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (self.width / 2 - text_main_rect[0] / 2, int(self.height * 0.15)),
                text_main_rect,
            ),
            text="Options Menu",
            manager=self.manager,
            visible=False,
            object_id="#options_menu",
        )

        email_rect = (450, 30)
        self.text_name_email_1 = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (
                    self.width - email_rect[0] - self.width * 0.01,
                    self.height - email_rect[1] * 3,
                ),
                email_rect,
            ),
            text="by Jungmoo Cheon (cjung-mo@student.42.fr)",
            manager=self.manager,
            object_id="#cjung-mo",
        )

        self.text_name_email_2 = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (
                    self.width - email_rect[0] - self.width * 0.01,
                    self.height - email_rect[1] * 2,
                ),
                email_rect,
            ),
            text="Sungyong Cho (sucho@student.42.fr)",
            manager=self.manager,
            object_id="#sucho",
        )

    def show_main_menu(self):
        self.text_main.show()
        self.button_to_singleplay.show()
        self.button_to_multiplay.show()
        self.button_to_options.show()

    def hide_main_menu(self):
        self.text_main.hide()
        self.button_to_singleplay.hide()
        self.button_to_multiplay.hide()
        self.button_to_options.hide()

    def show_options_menu(self):
        self.text_options.show()
        self.capture_stone_text.show()
        self.capture_stone_label.show()
        self.capture_stone_slider.show()
        self.advantage_text.show()
        self.black_capture_label.show()
        self.black_capture_slider.show()
        self.white_capture_label.show()
        self.white_capture_slider.show()
        self.capture_enable_text.show()
        self.capture_button.show()
        self.doublethree_text_label.show()
        self.doublethree_button.show()
        self.dropdown_label.show()
        self.dropdown_menu.show()
        self.button_to_main.show()

    def hide_options_menu(self):
        self.text_options.hide()
        self.capture_stone_text.hide()
        self.capture_stone_label.hide()
        self.capture_stone_slider.hide()
        self.advantage_text.hide()
        self.black_capture_label.hide()
        self.black_capture_slider.hide()
        self.white_capture_label.hide()
        self.white_capture_slider.hide()
        self.capture_enable_text.hide()
        self.capture_button.hide()
        self.doublethree_text_label.hide()
        self.doublethree_button.hide()
        self.dropdown_label.hide()
        self.dropdown_menu.hide()
        self.button_to_main.hide()

    def draw(self):
        self.screen.fill(WHITE)  # Set a background color for the screen

        if self.menu == MAIN_MENU:
            self.show_main_menu()
            self.hide_options_menu()

        elif self.menu == OPTIONS_MENU:
            self.show_options_menu()
            self.hide_main_menu()

        # TODO: responsive
        # self.screen.blit(self.text_name_email_1, (self.width - 550, self.height - 70))
        # self.screen.blit(self.text_name_email_2, (self.width - 460, self.height - 40))
        self.manager.draw_ui(self.screen)
        pygame.display.update()

    def wait_for_key(self):
        running = True
        clock = pygame.time.Clock()

        selected_options = {
            "capture": True,
            "mode": None,
            "doublethree": True,
            "option": self.dropdown_menu.selected_option,
            "capture_limit": 10,
            "advantage_black": 0,
            "advantage_white": 0,
        }
        while running:
            time_delta = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    # running = False

                self.manager.process_events(event)

                if event.type == pygame.USEREVENT:
                    if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                        if event.ui_element == self.button_to_main:
                            self.menu = MAIN_MENU

                        if event.ui_element == self.button_to_options:
                            self.menu = OPTIONS_MENU

                        if event.ui_element == self.button_to_singleplay:
                            selected_options["mode"] = "single"
                            self.menu = None
                            running = False
                        if event.ui_element == self.button_to_multiplay:
                            selected_options["mode"] = "double"
                            # TODO: append all the options and return
                            self.menu = None
                            running = False

                        if self.menu == OPTIONS_MENU:
                            if event.ui_element == self.capture_button:
                                if self.capture_button.text == "Enable":
                                    self.capture_button.set_text("Disable")
                                    selected_options["capture"] = False
                                else:
                                    self.capture_button.set_text("Enable")
                                    selected_options["capture"] = True
                            if event.ui_element == self.doublethree_button:
                                if self.doublethree_button.text == "Enable":
                                    self.doublethree_button.set_text("Disable")
                                    selected_options["doublethree"] = False
                                else:
                                    self.doublethree_button.set_text("Enable")
                                    selected_options["doublethree"] = True
                            if event.ui_element == self.dropdown_menu:
                                selected_options[
                                    "option"
                                ] = self.dropdown_menu.selected_option

                elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                    if event.ui_element == self.capture_stone_slider:
                        value = int(self.capture_stone_slider.get_current_value())
                        selected_options["capture_limit"] = value
                        self.capture_stone_label.set_text(str(value))

                        # Update the maximum value of the second slider dynamically
                        slider2_max_value = value / 2 + (
                            1 if (value / 2) % 2 != 0 else 0
                        )
                        self.black_capture_slider.value_range = (
                            0,
                            slider2_max_value,
                        )
                        self.white_capture_slider.value_range = (
                            0,
                            slider2_max_value,
                        )
                        # Adjust the position of slider2 within the new range
                        # current_value = self.black_capture_slider.get_current_value()
                        # if current_value > slider2_max_value:
                        #     self.black_capture_slider.set_current_value(0)
                        #     self.white_capture_slider.set_current_value(0)
                        self.black_capture_label.set_text(str(0))
                        self.black_capture_slider.set_current_value(0)
                        self.white_capture_label.set_text(str(0))
                        self.white_capture_slider.set_current_value(0)

                    elif event.ui_element == self.black_capture_slider:
                        selected_options["advantage_black"] = int(
                            self.black_capture_slider.get_current_value()
                        )
                        self.black_capture_label.set_text(
                            str(int(self.black_capture_slider.get_current_value()))
                        )
                    elif event.ui_element == self.white_capture_slider:
                        selected_options["advantage_white"] = int(
                            self.white_capture_slider.get_current_value()
                        )
                        self.white_capture_label.set_text(
                            str(int(self.white_capture_slider.get_current_value()))
                        )

            self.manager.update(time_delta)
            self.draw()
        return selected_options

import pygame
import pygame_gui


class ModalWindow:
    def __init__(self, manager, screen_size):
        self.manager = manager
        self.width = screen_size[0]
        self.height = screen_size[1]
        self.is_open = False

        panel_width = self.width / 3
        panel_height = 2 * self.height / 5
        panel_x = (self.width - panel_width) // 2
        panel_y = (self.height - panel_height) // 2

        self.modal_window = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((panel_x, panel_y), (panel_width, panel_height)),
            manager=self.manager,
            visible=False,
        )

        label_width = 2 * panel_width // 3
        label_x = (panel_width - label_width) // 2
        label_y = 30

        self.modal_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((label_x, label_y), (label_width, 40)),
            text="This is a modal window!",
            manager=self.manager,
            container=self.modal_window,
        )

        button_width = panel_width // 3
        button_height = panel_height / 10
        button_padding = button_width / 5  # Padding from the bottom of the panel

        button_x_left = (panel_width / 2) - button_width - button_padding
        button_y = 3 * panel_height / 4

        self.back_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (button_x_left, button_y), (button_width, button_height)
            ),
            text="Back to Main",
            manager=self.manager,
            container=self.modal_window,
        )

        button_x_right = panel_width / 2
        self.exit_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (button_x_right, button_y), (button_width, button_height)
            ),
            text="Exit",
            manager=self.manager,
            container=self.modal_window,
        )

    def open_modal(self):
        self.modal_window.show()
        self.is_open = True
        self.modal_window.visible = True

    def close_modal(self):
        self.modal_window.hide()
        self.is_open = False

    def set_modal_message(self, str):
        self.modal_label.set_text(str)

    def wait_for_response(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.exit_button:
                        pygame.quit()
                    elif event.ui_element == self.back_button:
                        self.close_modal()
                        return True
            self.manager.process_events(event)

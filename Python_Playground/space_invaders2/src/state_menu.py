import pygame

from .state import State
from .state_test import TestState
from .functions import draw_text, draw_lines
from .settings import WIDTH, HEIGHT

Y_GAP = 50
KEY_PRESS_DELAY = 0.1


class MenuState(State):
    def __init__(self, game) -> None:
        super().__init__(game)
        self.selected = None
        self.key_down = False
        self.key_timer = 0

        self.points1 = [
            (WIDTH // 2 - 100, HEIGHT // 4),
            (WIDTH // 2 + 100, HEIGHT // 4),
            (WIDTH // 2 + 100, HEIGHT // 4 + 50),
            (WIDTH // 2 - 100, HEIGHT // 4 + 50),
            (WIDTH // 2 - 100, HEIGHT // 4),
        ]
        self.points2 = [
            (WIDTH // 2 - 100, HEIGHT // 4 + 25),
            (WIDTH // 2 - 200, HEIGHT // 4 + 25),
            (WIDTH // 2 - 200, 3 * HEIGHT // 4),
            (WIDTH // 2 + 200, 3 * HEIGHT // 4),
            (WIDTH // 2 + 200, HEIGHT // 4 + 25),
            (WIDTH // 2 + 100, HEIGHT // 4 + 25),
        ]
        self.menu_title_position = (WIDTH // 2, HEIGHT // 4 + 25)
        self.options = ["New Game", "Options", "Exit"]
        self.options_start_pos = HEIGHT // 2 - 50

    def update(self, dt: float, **kwargs) -> None:
        """Update game object in game loop

        Args:
            dt (float): time since last frame
        """
        # Prevent rapid changing of menu selection
        if self.key_down:
            self.key_timer += dt
            if self.key_timer > KEY_PRESS_DELAY:
                self.key_down = False
                self.key_timer = 0
            return

        keys = pygame.key.get_pressed()
        # Change menu selection
        if keys[pygame.K_UP]:
            self.key_down = True
            if self.selected == None:
                self.selected = len(self.options) - 1
            else:
                self.selected -= 1
                if self.selected < 0:
                    self.selected = len(self.options) - 1
        elif keys[pygame.K_DOWN]:
            self.key_down = True
            if self.selected == None:
                self.selected = 0
            else:
                self.selected += 1
                if self.selected >= len(self.options):
                    self.selected = 0

        # Select option
        if keys[pygame.K_RETURN]:
            # Start game
            if self.selected == 0:
                self.should_exit = True
            # Options
            if self.selected == 1:
                pass
            # Exit
            if self.selected == 2:
                pygame.quit()
                quit()

    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """
        draw_lines(window, self.points1, 4)
        draw_lines(window, self.points2, 4)
        draw_text(window, "Menu", *self.menu_title_position, self.game.assets["font"])
        self._draw_options(window)

    def enter(self, **kwargs) -> None:
        """Actions to perform upon entering the state"""
        pass

    def exit(self) -> None:
        """Actions to perform upon exiting the state"""

        self.next_state = TestState(self.game)

    def process_event(self, event: pygame.event.Event):
        """Handle specific event

        Args:
            event (pygame.event.Event): event to handle
        """

        pass

    def _draw_options(self, window: pygame.Surface) -> None:
        """Draws the menu options to the screen

        Args:
            window (pygame.Surface): pygame surface to draw on
        """
        for index, option in enumerate(self.options):
            draw_text(
                window,
                option,
                WIDTH // 2,
                self.options_start_pos + Y_GAP * index,
                self.game.assets["font"],
                self.selected != None and self.selected == index,
            )

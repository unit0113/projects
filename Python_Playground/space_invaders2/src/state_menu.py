import pygame

from .state import State
from .state_menu_ship_select_transition import MenuShipSelectTransitionState
from .state_ship_select import ShipSelectState
from .functions import create_stacked_text, draw_lines
from .text import Text
from .settings import WIDTH, HEIGHT, KEY_PRESS_DELAY


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

        self.title_text = Text(
            "Menu", (WIDTH // 2, HEIGHT // 4 + 25), self.game.assets["font"]
        )
        self.menu_title_position = (WIDTH // 2, HEIGHT // 4 + 25)
        self.options = create_stacked_text(
            ["New Game", "Options", "Exit"],
            WIDTH // 2,
            HEIGHT // 2 - 50,
            50,
            self.game.assets["font"],
        )

        # Semi-transparent backround
        # Half height due to overlap with main rect
        self.title_backround_rect = pygame.Surface(
            (
                self.points1[1][0] - self.points1[0][0],
                (self.points1[2][1] - self.points1[0][1]) // 2,
            )
        )
        self.title_backround_rect.set_alpha(128)
        self.title_backround_rect.fill((0, 0, 0))

        self.main_backround_rect = pygame.Surface(
            (
                self.points2[3][0] - self.points2[1][0],
                self.points2[2][1] - self.points2[0][1],
            )
        )
        self.main_backround_rect.set_alpha(128)
        self.main_backround_rect.fill((0, 0, 0))

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

        # Mouse over
        pos = pygame.mouse.get_pos()
        for index, option in enumerate(self.options):
            if option.mouse_over(pos):
                self.selected = index
                break

        # Select option
        if keys[pygame.K_RETURN]:
            self._select_menu_option(self.selected)

    def _select_menu_option(self, selected: int = None) -> None:
        """Executes a selected menu option

        Args:
            selected (int): Index of selected option
        """
        # Start game
        if selected == 0:
            self.should_exit = True
        # Options
        if selected == 1:
            pass
        # Exit
        if selected == 2:
            pygame.quit()
            quit()

    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """
        # Draw transparent rects
        window.blit(self.title_backround_rect, self.points1[0])
        window.blit(self.main_backround_rect, self.points2[1])

        # Draw outline
        draw_lines(window, self.points1, 4)
        draw_lines(window, self.points2, 4)

        # Draw text
        self.title_text.draw(window)
        for index, option in enumerate(self.options):
            option.draw(window, self.selected == index)

    def enter(self, **kwargs) -> None:
        """Actions to perform upon entering the state"""
        pass

    def exit(self) -> None:
        """Actions to perform upon exiting the state"""

        self.next_state = MenuShipSelectTransitionState(
            self.game,
            self.points1,
            self.points2,
            self.title_backround_rect,
            self.main_backround_rect,
            self.title_text,
            self.options,
        )

    def process_events(self, events: list[pygame.event.Event]):
        """Handle game events

        Args:
            events (list[pygame.event.Event]): events to handle
        """

        # Select option mouse
        for event in events:
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                for index, option in enumerate(self.options):
                    if option.mouse_over(pos):
                        self._select_menu_option(index)
                        break
                break

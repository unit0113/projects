import pygame
import pytweening as tween

from .state import State
from .state_ship_select import ShipSelectState
from .functions import draw_lines
from .text import Text
from .button import Button
from .settings import WIDTH, HEIGHT

TEXT_TWEEN_DURATION = 0.3
LINE_TWEEN_DURATION = 0.75
LINE_TWEEN_DELAY = 0.5
SHIP_SELECT_TEXT_START = LINE_TWEEN_DELAY + LINE_TWEEN_DURATION + 0.25
SHIP_SELECT_TEXT_DELAY = 0.1
BUTTON_TWEEN_TIME = 0.4
FINAL_DELAY = 0.0


class MenuShipSelectTransitionState(State):
    def __init__(
        self,
        game,
        points1: list[tuple[int, int]],
        points2: list[tuple[int, int]],
        title_rect: pygame.Rect,
        main_rect: pygame.Rect,
        menu_title_text: Text,
        menu_options: list[Text],
    ) -> None:
        super().__init__(game)
        self.points1 = points1
        self.drawpoints1 = points1[:]
        self.points2 = points2
        self.drawpoints2 = points2[:]
        self.title_rect = title_rect
        self.main_rect = main_rect

        # Set next state and extract target points
        self.next_state = ShipSelectState(game)
        self.points1_tgt = self.next_state.points1
        self.points2_tgt = self.next_state.points2
        self.ship_select_characteristics = self.next_state.characteristics

        # Menu text objects
        self.menu_title_text = menu_title_text
        self.menu_options = menu_options

        # Initilize timers
        self.timer = 0

        # Tween variables
        self.line_tween_complete = False
        self.menu_alpha = 1
        self.ship_select_fade_start_times = [
            SHIP_SELECT_TEXT_START + index * SHIP_SELECT_TEXT_DELAY
            for index in range(len(self.ship_select_characteristics))
        ]
        self.ship_select_alphas = [0 for _ in self.ship_select_characteristics]
        self.button_alpha = 0
        self.button_fade_start_time = self.ship_select_fade_start_times[
            4 * len(self.ship_select_fade_start_times) // 5
        ]

        # Backgrounds
        self.title_backround_rect = pygame.Surface(
            (
                self.points1[1][0] - self.points1[0][0],
                (self.points1[2][1] - self.points1[0][1]) // 2,
            )
        )
        self.title_backround_rect.set_alpha(128)
        self.title_backround_rect.fill((0, 0, 0))

        self.title_background_start_dimensions = (
            self.title_backround_rect.get_width(),
            self.title_backround_rect.get_height(),
        )
        self.title_background_target_dimensions = (
            self.points1_tgt[1][0] - self.points1_tgt[0][0],
            (self.points1_tgt[2][1] - self.points1_tgt[0][1]) // 2,
        )

        self.main_backround_rect = pygame.Surface(
            (
                self.points2[3][0] - self.points2[1][0],
                self.points2[2][1] - self.points2[0][1],
            )
        )
        self.main_backround_rect.set_alpha(128)
        self.main_backround_rect.fill((0, 0, 0))

        self.main_background_start_dimensions = (
            self.main_backround_rect.get_width(),
            self.main_backround_rect.get_height(),
        )
        self.main_background_target_dimensions = (
            self.points2_tgt[3][0] - self.points2_tgt[1][0],
            self.points2_tgt[2][1] - self.points2_tgt[0][1],
        )

        # Ship select button
        self.select_button = Button(
            WIDTH // 2, HEIGHT - 375, "Select Ship", self.game.assets["font"], 310
        )

        # Ship select status bars
        self.status_bars = self.next_state.status_bars

    def update(self, dt: float, **kwargs) -> None:
        """Update game object in game loop

        Args:
            dt (float): time since last frame
        """
        # Tween menu text
        self.timer += dt
        self.menu_alpha = (
            0
            if self.timer >= TEXT_TWEEN_DURATION
            else 1 - tween.easeInOutSine(self.timer / TEXT_TWEEN_DURATION)
        )

        # Tween lines and background
        if self.timer > LINE_TWEEN_DELAY:
            if self.timer < LINE_TWEEN_DELAY + LINE_TWEEN_DURATION:
                tween_fraction = (self.timer - LINE_TWEEN_DELAY) / LINE_TWEEN_DURATION
                # Lines
                self.drawpoints1 = [
                    self._tween_points(start, target, tween_fraction)
                    for start, target in zip(self.points1, self.points1_tgt)
                ]
                self.drawpoints2 = [
                    self._tween_points(start, target, tween_fraction)
                    for start, target in zip(self.points2, self.points2_tgt)
                ]
                # Backgrounds
                self.title_backround_rect = pygame.transform.scale(
                    self.title_backround_rect,
                    (
                        self._tween_points(
                            self.title_background_start_dimensions,
                            self.title_background_target_dimensions,
                            tween_fraction,
                        )
                    ),
                )
                self.main_backround_rect = pygame.transform.scale(
                    self.main_backround_rect,
                    (
                        self._tween_points(
                            self.main_background_start_dimensions,
                            self.main_background_target_dimensions,
                            tween_fraction,
                        )
                    ),
                )

            # If line/background tweening complete, set to target values
            else:
                if not self.line_tween_complete:
                    self.line_tween_complete = True
                    self.drawpoints1 = self.points1_tgt
                    self.drawpoints2 = self.points2_tgt
                    self.title_backround_rect = pygame.transform.scale(
                        self.title_backround_rect,
                        self.title_background_target_dimensions,
                    )
                    self.main_backround_rect = pygame.transform.scale(
                        self.main_backround_rect, self.main_background_target_dimensions
                    )

        # Tween ship select text
        for index, start_time in enumerate(self.ship_select_fade_start_times):
            if self.timer <= start_time:
                self.ship_select_alphas[index] = 0
            elif self.timer >= start_time + TEXT_TWEEN_DURATION:
                self.ship_select_alphas[index] = 1
            else:
                self.ship_select_alphas[index] = tween.easeInOutSine(
                    (self.timer - start_time) / TEXT_TWEEN_DURATION
                )

        # Tween button
        if self.timer > self.button_fade_start_time:
            self.button_alpha = (
                1
                if self.timer > self.button_fade_start_time + BUTTON_TWEEN_TIME
                else tween.easeInOutSine(
                    (self.timer - self.button_fade_start_time) / BUTTON_TWEEN_TIME
                )
            )

        # Check if complete
        if (
            self.timer
            > self.ship_select_fade_start_times[-1] + TEXT_TWEEN_DURATION + FINAL_DELAY
        ):
            self.should_exit = True

    def _tween_points(
        self, start: tuple[int, int], target: tuple[int, int], fraction: float
    ) -> tuple[int, int]:
        """Function to calculate the tweening between two points

        Args:
            start (tuple[int, int]): starting point
            target (tuple[int, int]): target point
            fraction (float): percentage of tween completed

        Returns:
            tuple[int, int]: current point
        """
        if fraction >= 1:
            return target

        x = tween.easeInOutSine(fraction) * (target[0] - start[0]) + start[0]
        y = tween.easeInOutSine(fraction) * (target[1] - start[1]) + start[1]
        return (x, y)

    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """
        # Draw transparent rects
        window.blit(self.title_backround_rect, self.drawpoints1[0])
        window.blit(self.main_backround_rect, self.drawpoints2[1])

        # Draw outline
        draw_lines(window, self.drawpoints1, 4)
        draw_lines(window, self.drawpoints2, 4)

        # Draw menu text
        self.menu_title_text.draw(window, False, self.menu_alpha)
        for index, option in enumerate(self.menu_options):
            option.draw(window, index == 0, self.menu_alpha)

        # Draw ship select text
        status_bar_index = 0
        for index, (text, alpha) in enumerate(
            zip(self.ship_select_characteristics, self.ship_select_alphas)
        ):
            text.draw(window, False, alpha)
            # Draw status bars
            if index not in [0, 3, 7]:
                self.status_bars[status_bar_index].draw_empty(window, alpha)
                status_bar_index += 1

        # Draw button
        self.select_button.draw(window, self.button_alpha)

    def enter(self, **kwargs) -> None:
        """Actions to perform upon entering the state"""
        pass

    def exit(self) -> None:
        """Actions to perform upon exiting the state"""

        pass

    def process_events(self, events: list[pygame.event.Event]):
        """Handle game events

        Args:
            events (list[pygame.event.Event]): events to handle
        """

        pass

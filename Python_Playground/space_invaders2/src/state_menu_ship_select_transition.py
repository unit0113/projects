import pygame
import pytweening as tween

from .state import State
from .state_ship_select import ShipSelectState
from .functions import draw_lines
from .text import Text
from .settings import WIDTH, HEIGHT

LINE_TWEEN_DURATION = 0.5
LINE_TWEEN_DELAY = 1


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

        # Menu text objects
        self.menu_title_text = menu_title_text
        self.menu_options = menu_options

        # Initilize timers
        self.line_timer = 0

    def update(self, dt: float, **kwargs) -> None:
        """Update game object in game loop

        Args:
            dt (float): time since last frame
        """

        self.line_timer += dt
        if self.line_timer > LINE_TWEEN_DELAY:
            if self.line_timer < LINE_TWEEN_DELAY + LINE_TWEEN_DURATION:
                self.drawpoints1 = [
                    self._tween_points(
                        start,
                        target,
                        (self.line_timer - LINE_TWEEN_DELAY) / LINE_TWEEN_DURATION,
                    )
                    for start, target in zip(self.points1, self.points1_tgt)
                ]
                self.drawpoints2 = [
                    self._tween_points(
                        start,
                        target,
                        (self.line_timer - LINE_TWEEN_DELAY) / LINE_TWEEN_DURATION,
                    )
                    for start, target in zip(self.points2, self.points2_tgt)
                ]
            else:
                self.drawpoints1 = self.points1_tgt
                self.drawpoints2 = self.points2_tgt

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

        x = tween.easeOutSine(fraction) * (target[0] - start[0]) + start[0]
        y = tween.easeOutSine(fraction) * (target[1] - start[1]) + start[1]
        return (x, y)

    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        draw_lines(window, self.drawpoints1, 4)
        draw_lines(window, self.drawpoints2, 4)

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

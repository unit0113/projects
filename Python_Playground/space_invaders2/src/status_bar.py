import pygame
from math import ceil
from .settings import GREY, RED_YELLOW_BLUE

OUTER_BORDER_RADIUS = 5
INNER_BORDER_RADIUS = 3
OUTER_WIDTH = 2
MAX_PIPS = len(RED_YELLOW_BLUE)


class StatusBar:
    tween_fractions = [
        (max(0, (index - 1) / MAX_PIPS - (1 / (2 * MAX_PIPS))), index / MAX_PIPS)
        for index in range(1, MAX_PIPS + 1)
    ]

    def __init__(
        self,
        pos: tuple[int, int],
        size: tuple[int, int],
        min_val: float,
        max_val: float,
    ) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.pos = pos
        self.size = size

        self.outer_rect = pygame.Rect(
            pos[0],
            pos[1] - size[1] // 2,
            *size,
        )

        # Alpha surface
        self.alpha_surface = pygame.Surface(size)
        self.outer_alpha_rect = pygame.Rect(
            0,
            0,
            *size,
        )
        self.alpha_pos = (
            pos[0],
            pos[1] - size[1] // 2,
        )

        # Alpha pips
        self.alpha_pip_surf = pygame.Surface(
            (
                (self.size[0] - 2 * OUTER_WIDTH) / MAX_PIPS + 1,
                self.size[1] - 2 * OUTER_WIDTH,
            )
        )

    def _get_percent(self, val: float) -> float:
        """Determines the percentage value of the input based on the range of possible values

        Returns:
            float: Percentage of the value in the range
        """
        return (val - self.min_val) / (self.max_val - self.min_val)

    def update_value(self, new_val: float) -> None:
        """Updates the held value

        Args:
            new_val (float): New value
        """
        num_pips = max(1, ceil(self._get_percent(new_val) * MAX_PIPS))

        # Regular pips
        self.rects = []
        blip_width = (self.size[0] - 2 * OUTER_WIDTH) / MAX_PIPS
        start_x = self.pos[0] + OUTER_WIDTH
        blip_start = start_x
        for index in range(num_pips):
            blip_end = round(start_x + (1 + index) * blip_width)
            rect = pygame.Rect(
                blip_start,
                self.pos[1] - self.size[1] // 2 + OUTER_WIDTH,
                blip_width + 1,
                self.size[1] - 2 * OUTER_WIDTH,
            )
            self.rects.append(rect)
            blip_start = blip_end

    def draw(self, window: pygame.Surface) -> None:
        """Draw object to the pygame window

        Args:
            window (pygame.Surface): Pygame surface to draw on
        """
        # Draw blips
        for index, rect in enumerate(self.rects):
            pygame.draw.rect(window, RED_YELLOW_BLUE[index], rect, 2)
            window.fill(RED_YELLOW_BLUE[index], rect)

        # Draw outline
        pygame.draw.rect(
            window, GREY, self.outer_rect, OUTER_WIDTH, OUTER_BORDER_RADIUS
        )

    def draw_empty(self, window: pygame.Surface, alpha: float = 1) -> None:
        """Draws empty outline with alpha

        Args:
            window (pygame.Surface): Pygame surface to draw on
            alpha (float, optional): Alpha value for status bar. Defaults to 1.
        """
        pygame.draw.rect(
            self.alpha_surface,
            GREY,
            self.outer_alpha_rect,
            OUTER_WIDTH,
            OUTER_BORDER_RADIUS,
        )
        self.alpha_surface.set_alpha(int(255 * alpha))
        window.blit(self.alpha_surface, self.alpha_pos)

    def draw_tween(self, window: pygame.Surface, fraction: float) -> None:
        """Tween pips onto screen

        Args:
            window (pygame.Surface): Pygame surface to draw on
            fraction (float): Progress of tween
        """
        # If complete
        if fraction >= 1:
            self.draw(window)
            return

        # Draw blips
        for index, rect in enumerate(self.rects):
            # pygame.draw.rect(alpha_surf, RED_YELLOW_BLUE[index], self.alpha_pip_rect, 2)
            self.alpha_pip_surf.fill(RED_YELLOW_BLUE[index])
            self.alpha_pip_surf.set_alpha(
                int(255 * self._calc_alpha(*self.tween_fractions[index], fraction))
            )
            window.blit(self.alpha_pip_surf, rect)

        # Draw outline
        pygame.draw.rect(
            window, GREY, self.outer_rect, OUTER_WIDTH, OUTER_BORDER_RADIUS
        )

    def _calc_alpha(self, start: float, finish: float, current: float) -> float:
        """Calculates the alpha value for a pip

        Args:
            start (float): Start fraction of tween
            finsh (float): End fraction of tween
            current (float): Current tween fraction

        Returns:
            float: Alpha value (0-1)
        """
        if current < start:
            return 0
        elif current > finish:
            return 1
        else:
            return (current - start) / (finish - start)

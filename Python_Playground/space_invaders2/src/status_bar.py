import pygame
from math import ceil
from .settings import GREY, RED_YELLOW_BLUE

OUTER_BORDER_RADIUS = 5
INNER_BORDER_RADIUS = 3
OUTER_WIDTH = 2


class StatusBar:
    def __init__(
        self,
        pos: tuple[int, int],
        size: tuple[int, int],
        min_val: float,
        max_val: float,
        start_val: float,
    ) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.pos = pos
        self.size = size
        self.update_value(start_val)

        self.outer_rect = pygame.Rect(
            pos[0],
            pos[1] - size[1] // 2,
            *size,
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
        num_pips = max(1, ceil(self._get_percent(new_val) * len(RED_YELLOW_BLUE)))

        self.rects = []
        blip_width = (self.size[0] - 2 * OUTER_WIDTH) / len(RED_YELLOW_BLUE)
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
        # Draw blip
        for index, rect in enumerate(self.rects):
            pygame.draw.rect(window, RED_YELLOW_BLUE[index], rect, 2)
            window.fill(RED_YELLOW_BLUE[index], rect)
        # Draw outline
        pygame.draw.rect(
            window, GREY, self.outer_rect, OUTER_WIDTH, OUTER_BORDER_RADIUS
        )

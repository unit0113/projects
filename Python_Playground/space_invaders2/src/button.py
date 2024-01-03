import pygame

from .settings import MAGENTA, DARK_GREY, BLACK

HIGHLIGHT_REC_SIZE = 20
OUTER_REC_SIZE = 16
INNER_REC_SIZE = 12


class Button:
    def __init__(
        self,
        x: int,
        y: int,
        text: str,
        font: pygame.Font,
        size: int,
    ) -> None:
        self.text = font.render(text, 1, MAGENTA)

        # Upper left corner of text
        self.start_x = x - self.text.get_width() // 2
        self.start_y = y + self.text.get_height() // 2

        # Outline
        self.rect_outer = pygame.Rect(
            self.start_x - OUTER_REC_SIZE // 2,
            self.start_y - OUTER_REC_SIZE // 2,
            size + OUTER_REC_SIZE,
            self.text.get_height() + OUTER_REC_SIZE,
        )
        self.rect_inner = pygame.Rect(
            self.start_x - INNER_REC_SIZE // 2,
            self.start_y - INNER_REC_SIZE // 2,
            size + INNER_REC_SIZE,
            self.text.get_height() + INNER_REC_SIZE,
        )

        # Highlighted variables
        self.highlighted = False

    def set_highlight(self, highlighted: bool) -> None:
        """Sets highligh of button

        Args:
            highlighted (bool): Whether button should be highlighted
        """
        self.highlighted = highlighted

    def mouse_over(self, mouse_pos: tuple[float]) -> bool:
        """Check if mouse position is on this button

        Args:
            mouse_pos (tuple[float]): X, Y position that the mouse clicked on

        Returns:
            bool: Whether this button was clicked on
        """

        return self.rect_outer.collidepoint(mouse_pos)

    def draw(self, window: pygame.Surface) -> None:
        """Draw objects to the pygame window"""

        inner_rect_color = DARK_GREY if self.highlighted else BLACK

        pygame.draw.rect(window, MAGENTA, self.rect_outer, border_radius=10)
        pygame.draw.rect(window, inner_rect_color, self.rect_inner, border_radius=10)
        window.blit(self.text, (self.start_x, self.start_y))

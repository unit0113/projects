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

        # Outline
        self.rect_pos = (
            x - self.text.get_width() // 2 - OUTER_REC_SIZE // 2,
            y + self.text.get_height() // 2 - OUTER_REC_SIZE // 2,
        )
        self.text_offset = (OUTER_REC_SIZE // 2, OUTER_REC_SIZE // 2)

        self.collision_outer = pygame.Rect(
            *self.rect_pos,
            size + OUTER_REC_SIZE,
            self.text.get_height() + OUTER_REC_SIZE,
        )

        self.rect_outer = pygame.Rect(
            0,
            0,
            size + OUTER_REC_SIZE,
            self.text.get_height() + OUTER_REC_SIZE,
        )

        self.rect_inner = pygame.Rect(
            (OUTER_REC_SIZE - INNER_REC_SIZE) // 2,
            (OUTER_REC_SIZE - INNER_REC_SIZE) // 2,
            size + INNER_REC_SIZE,
            self.text.get_height() + INNER_REC_SIZE,
        )

        # Alpha surface
        self.alpha_surface = pygame.Surface(
            (size + OUTER_REC_SIZE, self.text.get_height() + OUTER_REC_SIZE)
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

        return self.collision_outer.collidepoint(mouse_pos)

    def draw(self, window: pygame.Surface, alpha: float = 1) -> None:
        """Draw objects to the pygame window"""

        inner_rect_color = DARK_GREY if self.highlighted else BLACK
        self.alpha_surface.fill((255, 255, 255))
        self.alpha_surface.set_colorkey((255, 255, 255))

        pygame.draw.rect(self.alpha_surface, MAGENTA, self.rect_outer, border_radius=10)
        pygame.draw.rect(
            self.alpha_surface,
            inner_rect_color,
            self.rect_inner,
            border_radius=10,
        )
        self.alpha_surface.blit(self.text, self.text_offset)

        self.alpha_surface.set_alpha(int(255 * alpha))
        window.blit(self.alpha_surface, self.rect_pos)

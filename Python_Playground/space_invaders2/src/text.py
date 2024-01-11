import pygame

from .settings import MAGENTA, GREY


class Text:
    def __init__(
        self,
        string: str,
        pos: tuple[int, int],
        font: pygame.Font,
        left_justified: bool = False,
        color: tuple[int, int, int] = GREY,
    ) -> None:
        self.pos = pos
        self.standard_text = font.render(string, True, color)
        self.selected_text = font.render(f"| {string} |", True, MAGENTA)
        self.text_rect = self.standard_text.get_rect()

        self.left_justified = left_justified
        self.was_selected = False
        self._set_rect_pos()

    def _set_rect_pos(self) -> None:
        """Sets position of text rect"""
        if self.left_justified:
            self.text_rect.midleft = self.pos
        else:
            self.text_rect.center = self.pos

    def draw(
        self, window: pygame.Surface, selected: bool = False, alpha: float = 1
    ) -> None:
        """Draws text to window

        Args:
            window (pygame.Surface): Screen to draw on
            selected (bool): Wether the text is highlighted. Defaults to False
            alpha (float): Alpha value for text (0-1)
        """
        if selected:
            if not self.was_selected:
                self.text_rect = self.selected_text.get_rect()
                self._set_rect_pos()
            text_surf = self.selected_text.copy()
        else:
            if self.was_selected:
                self.text_rect = self.standard_text.get_rect()
                self._set_rect_pos()
            text_surf = self.standard_text.copy()
        self.was_selected = selected
        # Draw with alpha
        alpha_surf = pygame.Surface(text_surf.get_size(), pygame.SRCALPHA)
        alpha_surf.fill((255, 255, 255, int(255 * alpha)))
        text_surf.blit(alpha_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        window.blit(text_surf, self.text_rect)

    def mouse_over(self, pos: tuple[int, int]) -> bool:
        """Determines whether mouse is over Text object

        Args:
            pos (tuple[int, int]): Mouse position

        Returns:
            bool: Whether mouse overlaps with object
        """
        return self.text_rect.collidepoint(pos)

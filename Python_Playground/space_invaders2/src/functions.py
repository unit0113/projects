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
        self.selected__text = font.render(f"| {string} |", True, MAGENTA)
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

    def draw(self, window: pygame.Surface, selected: bool = False) -> None:
        """Draws text to window

        Args:
            window (pygame.Surface): Screen to draw on
            selected (bool): Wether the text is highlighted. Defaults to False
        """
        if selected:
            if not self.was_selected:
                self.text_rect = self.selected__text.get_rect()
                self._set_rect_pos()
            window.blit(self.selected__text, self.text_rect)
        else:
            if self.was_selected:
                self.text_rect = self.standard_text.get_rect()
                self._set_rect_pos()
            window.blit(self.standard_text, self.text_rect)
        self.was_selected = selected

    def mouse_over(self, pos: tuple[int, int]) -> bool:
        """Determines whether mouse is over Text object

        Args:
            pos (tuple[int, int]): Mouse position

        Returns:
            bool: Whether mouse overlaps with object
        """
        return self.text_rect.collidepoint(pos)


def create_stacked_text(
    text_list: list[str],
    x: int,
    start_y: int,
    y_gap: int,
    font: pygame.Font,
    left_justified: bool = False,
) -> list[Text]:
    """Generate multiple vertically stacked text objects

    Args:
        text_list (list[str]): List of text to draw
        x (int): X position for all text options
        start_y (int): Y position to start drawing
        y_gap (int): Gap between options
        font (pygame.Font): Pygame font
        left_justified (bool, optional): Whether the text should be left justified. Defaults to False.

    Returns:
        list[Text]: List of generated text objects
    """
    texts = []
    for index, item in enumerate(text_list):
        texts.append(Text(item, (x, start_y + index * y_gap), font, left_justified))
    return texts


def create_mixed_stacked_text(
    text_list: list[str],
    x: int,
    start_y: int,
    y_gap: int,
    font_list: list[pygame.Font],
    color_list: list[tuple[int, int, int]],
    left_justified: bool = False,
) -> list[Text]:
    """Generate multiple vertically stacked text objects with different fonts/colors

    Args:
        text_list (list[str]): List of text to draw
        x (int): X position for all text options
        start_y (int): Y position to start drawing
        y_gap (int): Gap between options
        font_list (list[pygame.Font]): list of pygame fonts
        color_list (list[tuple[int, int, int]]): lists of colors
        left_justified (bool, optional): Whether the text should be left justified. Defaults to False.

    Returns:
        list[Text]: List of generated text objects
    """
    texts = []
    for index, (item, font, color) in enumerate(zip(text_list, font_list, color_list)):
        texts.append(
            Text(item, (x, start_y + index * y_gap), font, left_justified, color)
        )
    return texts


def draw_lines(
    window: pygame.Surface,
    points: list[tuple[int, int]],
    line_width: int,
    color: tuple[int, int, int] = GREY,
) -> None:
    """Draws lines to the screen

    Args:
        window (pygame.Surface): Screen to draw on
        points (list[tuple[int, int]]): List of points to draw lines between
        line_width (int): Width of lines to draw
        color (tuple[int, int, int], optional): Line color. Defaults to GREY.
    """

    if len(points) < 2:
        return

    for index in range(1, len(points)):
        pygame.draw.line(window, color, points[index - 1], points[index], line_width)

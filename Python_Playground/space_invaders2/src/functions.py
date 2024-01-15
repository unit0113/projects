import pygame
from random import choice

from .text import Text
from .settings import GREY, MAX_WALK


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


def random_walk(
    current_ship_pos: list[int, int], center_ship_pos: list[int, int]
) -> None:
    """Randomly moves ship sprite around"""
    x_move = choice([-1, 0, 1])
    y_move = choice([-1, 0, 1])
    x = max(
        center_ship_pos[0] - MAX_WALK,
        min(center_ship_pos[0] + MAX_WALK, current_ship_pos[0] + x_move),
    )
    y = max(
        center_ship_pos[1] - MAX_WALK,
        min(center_ship_pos[1] + MAX_WALK, current_ship_pos[1] + y_move),
    )
    return [x, y]

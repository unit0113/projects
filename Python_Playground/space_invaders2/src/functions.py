import pygame

DEFAULT_COLOR = (200, 200, 200)
SELECTED_COLOR = (100, 40, 80)


def draw_text(
    window: pygame.Surface,
    string: str,
    x: int,
    y: int,
    font: pygame.Font,
    selected: bool = False,
    color: tuple[int, int, int] = None,
) -> None:
    """Draws text to screen

    Args:
        window (pygame.Surface): Screen to draw on
        string (str): String to draw
        x (int): Center X position
        y (int): Center Y position
        font (pygame.Font): Pygame font
        selected (bool): Wether the text is highlighted. Defaults to False
        color (tuple[int, int, int]): Text color. Defaults to None
    """
    if not color:
        color = SELECTED_COLOR if selected else DEFAULT_COLOR
    text = font.render(string, True, color)
    text_rect = text.get_rect()
    text_rect.center = (x, y)
    window.blit(text, text_rect)

    if selected:
        draw_text(
            window,
            "|",
            text_rect.x - 20,
            text_rect.centery,
            font,
            False,
            SELECTED_COLOR,
        )
        draw_text(
            window,
            "|",
            text_rect.right + 20,
            text_rect.centery,
            font,
            False,
            SELECTED_COLOR,
        )


def draw_lines(
    window: pygame.Surface,
    points: list[tuple[int, int]],
    line_width: int,
    color: tuple[int, int, int] = DEFAULT_COLOR,
) -> None:
    """Draws lines to the screen

    Args:
        window (pygame.Surface): Screen to draw on
        points (list[tuple[int, int]]): List of points to draw lines between
        line_width (int): Width of lines to draw
        color (tuple[int, int, int], optional): Line color. Defaults to DEFAULT_COLOR.
    """

    if len(points) < 2:
        return

    for index in range(1, len(points)):
        pygame.draw.line(window, color, points[index - 1], points[index], line_width)

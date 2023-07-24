import pygame
from typing import Callable

from src.colors import BLACK, MENU_PURPLE, HIGHLIGHT_WHITE
from src.settings import BUTTON_SIZE

HIGHLIGHT_REC_SIZE = 24
OUTER_REC_SIZE = 20
INNER_REC_SIZE = 14


class Button:
    def __init__(self, window:pygame.surface.Surface, x: int, y: int, text: str, size: int=BUTTON_SIZE, text_size: int=20) -> None:
        self.name = text
        self.font = pygame.font.SysFont('verdana', text_size, bold=True)
        self.text = self.font.render(text, 1, MENU_PURPLE) 
        self.window = window

        # Upper left corner of text
        self.start_x = x // 2 - size // 2
        self.start_y = y + self.text.get_height() // 2

        # Outline
        self.rect_outer = pygame.Rect(self.start_x - OUTER_REC_SIZE // 2, self.start_y - OUTER_REC_SIZE // 2, size + OUTER_REC_SIZE, self.text.get_height() + OUTER_REC_SIZE)
        self.rect_inner = pygame.Rect(self.start_x - INNER_REC_SIZE // 2, self.start_y - INNER_REC_SIZE // 2, size + INNER_REC_SIZE, self.text.get_height() + INNER_REC_SIZE)

        # Highlighted variables
        self.highlighted = False
        self.rect_highlight = pygame.Rect(self.start_x - HIGHLIGHT_REC_SIZE // 2, self.start_y - HIGHLIGHT_REC_SIZE // 2, size + HIGHLIGHT_REC_SIZE, self.text.get_height() + HIGHLIGHT_REC_SIZE)

        # Tweening variables
        self.offset_x = 0
        self.offset_y = 0
        self.tween_fx = None
        self.start = 0
        self.duration = 0
        self.tween_x = False
        self.range = 0

    def set_tween(self, tween_fx: Callable[[float], float], start: float, duration: float, tween_x: bool, range: int) -> None:
        """Starts a new tweening operation on the button

        Args:
            tween_fx (Callable[[float], float]): Tween function
            start (float): Start time
            duration (float): Duration of tween
            tween_x (bool): If tween is occurring in the X direction. True for X movement, False for Y movement
            range (int): Distance to tween
        """

        self.tween_fx = tween_fx
        self.start = start
        self.duration = duration
        self.tween_x = tween_x
        self.range = range

    def is_highlighted(self) -> bool:
        """Check if button is currently highlighted

        Returns:
            bool: If button is highlighted
        """
        return self.highlighted

    def highlight(self) -> None:
        """Set button as highlighted
        """

        self.highlighted = True

    def dehighlight(self) -> None:
        """Dehighlight this button
        """

        self.highlighted = False

    def mouse_over(self, mouse_pos: tuple[float]) -> bool:
        """Check if mouse position is on this button

        Args:
            mouse_pos (tuple[float]): X, Y position that the mouse clicked on

        Returns:
            bool: Whether this button was clicked on
        """

        return self.rect_outer.collidepoint(mouse_pos)

    def update(self, timer: float) -> None:
        """Perform tweening operations and calculate the new positions for owned objects

        Args:
            timer (float): Time since start of game state
        """

        # Tween if active
        if self.tween_fx:
            if self.start < timer < self.start + self.duration:
                if self.tween_x:
                    self.offset_x = self.range * self.tween_fx((timer - self.start) / self.duration)
                else:
                    self.offset_y = self.range * self.tween_fx((timer - self.start) / self.duration)

            # End tween calculations and return to default state
            elif timer > self.start + self.duration:
                self.tween_fx = None
                if self.tween_x:
                    self.start_x += self.range
                else:
                    self.start_y += self.range
                self.offset_x = 0
                self.offset_y = 0

            # Move recs to new offset
            self.rect_highlight.x = self.start_x - HIGHLIGHT_REC_SIZE // 2 + self.offset_x
            self.rect_outer.x = self.start_x - OUTER_REC_SIZE // 2 + self.offset_x
            self.rect_inner.x = self.start_x - INNER_REC_SIZE // 2 + self.offset_x
            self.rect_highlight.y = self.start_y - HIGHLIGHT_REC_SIZE // 2 + self.offset_y
            self.rect_outer.y = self.start_y - OUTER_REC_SIZE // 2 + self.offset_y
            self.rect_inner.y = self.start_y - INNER_REC_SIZE // 2 + self.offset_y

    def draw(self) -> None:
        """Draw objects to the pygame window
        """

        if self.highlighted:
            pygame.draw.rect(self.window, HIGHLIGHT_WHITE, self.rect_highlight, border_radius=10)
            
        pygame.draw.rect(self.window, MENU_PURPLE, self.rect_outer, border_radius=10)
        pygame.draw.rect(self.window, BLACK, self.rect_inner, border_radius=10)
        self.window.blit(self.text, (self.start_x + self.offset_x, self.start_y + self.offset_y))
import pygame
import pytweening as tween
from typing import Callable

from src.colors import BLACK, MENU_PURPLE


OUTER_REC_SIZE = 20
INNER_REC_SIZE = 14
SIZE = 100

class Button:
    def __init__(self, x: int, y: int, text: str, size: int=SIZE) -> None:
        self.font = pygame.font.SysFont('verdana', 20, bold=True)
        self.text = self.font.render(text, 1, MENU_PURPLE) 

        self.start_x = x // 2 - size // 2
        self.start_y = y + size // 2

        self.rect_outer = pygame.Rect(self.start_x - OUTER_REC_SIZE // 2, self.start_y - OUTER_REC_SIZE // 2, size + OUTER_REC_SIZE, self.text.get_height() + OUTER_REC_SIZE)
        self.rect_inner = pygame.Rect(self.start_x - INNER_REC_SIZE // 2, self.start_y - INNER_REC_SIZE // 2, size + INNER_REC_SIZE, self.text.get_height() + INNER_REC_SIZE)

        # Tweening variables
        self.offset_x = 0
        self.offset_y = 0
        self.tween = None
        self.start = 0
        self.duration = 0
        self.tween_x = False
        self.range = 0

    def set_tween(self, tween_fx: Callable[[float], float], start: float, duration: float, tween_x: bool, range: int) -> None:
        self.tween = tween_fx
        self.start = start
        self.duration = duration
        self.tween_x = tween_x
        self.range = range

    def update(self, timer: float) -> None:
        # Tween if active
        if self.tween:
            if self.start < timer < self.start + self.duration:
                if self.tween_x:
                    self.offset_x = self.range * tween.easeInSine((timer - self.start) / (self.start + self.duration))  # Need to rethink
                else:
                    self.offset_y = self.range * tween.easeInSine((timer - self.start) / (self.start + self.duration))

            # End tween calculations and return to default state
            elif timer > self.start + self.duration:
                self.tween = None

            # Move recs to new offset
            self.rect_outer.x = self.start_x - OUTER_REC_SIZE // 2 + self.offset_x
            self.rect_inner.x = self.start_x - INNER_REC_SIZE // 2 + self.offset_x
            self.rect_outer.y = self.start_y - OUTER_REC_SIZE // 2 + self.offset_y
            self.rect_inner.y = self.start_y - INNER_REC_SIZE // 2 + self.offset_y

    def draw(self, window: pygame.surface.Surface) -> None:
        pygame.draw.rect(window, MENU_PURPLE, self.rect_outer, border_radius=10)
        pygame.draw.rect(window, BLACK, self.rect_inner, border_radius=10)
        window.blit(self.text, (self.start_x + self.offset_x, self.start_y + self.offset_y))

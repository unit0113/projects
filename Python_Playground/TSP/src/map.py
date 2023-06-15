import pygame
from typing import Callable


class Map:
    def __init__(self, x, y) -> None:
        self.img = pygame.image.load(r'assets\florida.jpg').convert_alpha()
        self.img.set_alpha(0)
        self.x = x
        self.y = y

        # Tween vaiables
        self.start = 0
        self.duration = 0
        self.tween_fx = None

    def set_fade_tween(self, tween_fx: Callable[[float], float], start: float, duration: float) -> None:
        self.tween_fx = tween_fx
        self.start = start
        self.duration = duration

    def update(self, timer):
        if self.start < timer < self.start + self.duration:
            self.img.set_alpha(255 * self.tween_fx((timer - self.start) / (self.start + self.duration)))

    def draw(self, window):
        window.blit(self.img, (self.x, self.y))
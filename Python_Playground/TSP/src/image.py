import pygame
from typing import Callable


class Image:
    def __init__(self, window: pygame.surface.Surface, x, y) -> None:
        self.img = pygame.image.load(r'assets\florida.jpg').convert_alpha()
        self.img.set_alpha(0)
        self.orig_img = self.img
        self.start_height = self.img.get_height()
        self.start_width = self.img.get_width()
        self.x = x
        self.y = y
        self.right_x = x + self.start_width
        self.bottom_y = y + self.start_height
        self.window = window

        # Tween vaiables
        self.start = 0
        self.duration = 0
        self.fade_tween_fx = None
        self.enlarge_tween_fx = None
        self.shrink_tween_fx = None

    def set_fade_tween(self, tween_fx: Callable[[float], float], start: float, duration: float) -> None:
        self.fade_tween_fx = tween_fx
        self.start = start
        self.duration = duration

    def set_enlarge_tween(self, tween_fx: Callable[[float], float], start: float, duration: float) -> None:
        self.enlarge_tween_fx = tween_fx
        self.start = start
        self.duration = duration

    def set_shrink_tween(self, tween_fx: Callable[[float], float], start: float, duration: float) -> None:
        self.shrink_tween_fx = tween_fx
        self.start = start
        self.duration = duration

    def get_x_y_height_width(self) -> tuple[int, int, int, int]:
        """ Provides the position and dimensions of the image

        Returns:
            tuple[int, int, int, int]: position and dimensions of the image
        """

        return self.x, self.y, self.start_height, self.start_width

    def update(self, timer):
        # Fade in tweening
        if self.fade_tween_fx:
            if self.start < timer < self.start + self.duration:
                self.img.set_alpha(255 * self.fade_tween_fx((timer - self.start) / self.duration))
            elif timer > self.start + self.duration:
                self.fade_tween_fx = None

        # Enlarging tweening
        if self.enlarge_tween_fx:
            if self.start < timer < self.start + self.duration:
                tween_val = self.enlarge_tween_fx((timer - self.start) / self.duration)
                self.img = pygame.transform.smoothscale(self.orig_img, (self.start_width + 200 * tween_val, self.start_height + 150 * tween_val))
                self.x = self.right_x - self.img.get_width()
                self.y = self.bottom_y - self.img.get_height()
            elif timer > self.start + self.duration:
                self.enlarge_tween_fx = None
                # Save current width, height for next tweening
                self.start_height = self.img.get_height()
                self.start_width = self.img.get_width()

        # Shrinking tweening
        if self.shrink_tween_fx:
            if self.start < timer < self.start + self.duration:
                tween_val = self.shrink_tween_fx((timer - self.start) / self.duration)
                self.img = pygame.transform.smoothscale(self.orig_img, (self.start_width - 200 * tween_val, self.start_height - 150 * tween_val))
                self.x = self.right_x - self.img.get_width()
                self.y = self.bottom_y - self.img.get_height()
            elif timer > self.start + self.duration:
                self.shrink_tween_fx = None
                # Save current width, height for next tweening
                self.start_height = self.img.get_height()
                self.start_width = self.img.get_width()

        

    def draw(self):
        self.window.blit(self.img, (self.x, self.y))
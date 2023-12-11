import pygame
import random

from .behavior import Behavior


class RandomDoubleTapFireBehavior(Behavior):
    def __init__(self, cooldown: int) -> None:
        super().__init__()
        self.cooldown = cooldown
        self.waiting_on_primary = True
        self.shot_spacing = 50
        self.next_shot = self.get_next_shot_time()

    def get_next_shot_time(self) -> int:
        if self.waiting_on_primary:
            return (
                pygame.time.get_ticks()
                + random.randint(self.cooldown // 2, self.cooldown)
                + self.cooldown
            )
        else:
            return pygame.time.get_ticks() + self.shot_spacing

    def update(self, dt: float):
        if pygame.time.get_ticks() > self.next_shot:
            self._can_fire = True

    def fire(self):
        self._can_fire = False
        self.waiting_on_primary = not self.waiting_on_primary
        self.next_shot = self.get_next_shot_time()

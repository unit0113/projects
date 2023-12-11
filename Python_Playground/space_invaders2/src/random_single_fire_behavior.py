import pygame
import random

from .behavior import Behavior


class RandomSingleFireBehavior(Behavior):
    def __init__(self, cooldown: int) -> None:
        super().__init__()
        self.cooldown = cooldown
        self.next_shot = self.get_next_shot_time()

    def get_next_shot_time(self) -> int:
        return (
            pygame.time.get_ticks()
            + random.randint(self.cooldown // 2, self.cooldown)
            + self.cooldown
        )

    def update(self, dt: float):
        if pygame.time.get_ticks() > self.next_shot:
            self._can_fire = True

    def fire(self):
        self._can_fire = False
        self.next_shot = self.get_next_shot_time()

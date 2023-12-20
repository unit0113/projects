import pygame
import random

from .behavior import Behavior


class BeamFireBehavior(Behavior):
    def __init__(self, cooldown: float, duration: float) -> None:
        super().__init__()
        self.cooldown = cooldown
        self.duration = duration
        self.next_shot = self.get_next_shot_time()
        self.start_of_shot = 0

    def get_next_shot_time(self) -> int:
        return (
            pygame.time.get_ticks()
            + random.randint(self.cooldown // 2, self.cooldown)
            + self.cooldown
            + self.duration
        )

    def update(self, dt: float):
        if pygame.time.get_ticks() > self.next_shot:
            self.start_of_shot = self.last_shot = pygame.time.get_ticks()
            self._can_fire = True
        elif pygame.time.get_ticks() < self.start_of_shot + self.duration:
            self.last_shot = pygame.time.get_ticks()
            self._can_fire = True
        else:
            self._can_fire = False

    def fire(self):
        self._can_fire = False
        self.next_shot = self.get_next_shot_time()

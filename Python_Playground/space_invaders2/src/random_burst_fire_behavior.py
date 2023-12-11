import pygame
import random

from .behavior import Behavior


class RandomBurstFireBehavior(Behavior):
    def __init__(self, cooldown: int, burst_size: int) -> None:
        super().__init__()
        self.cooldown = cooldown
        self.burst_size = burst_size
        self.shot_counter = 0
        self.shot_spacing = 50
        self.next_shot = self.get_next_shot_time()

    def get_next_shot_time(self) -> int:
        if not self.shot_counter:
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
        self.shot_counter += 1
        if self.shot_counter >= self.burst_size:
            self.shot_counter = 0
        self.next_shot = self.get_next_shot_time()

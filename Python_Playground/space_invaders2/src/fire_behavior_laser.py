import pygame
import random

from .behavior import Behavior


class LaserFireBehavior(Behavior):
    def __init__(self, cooldown: int, burst_size: int = 1) -> None:
        super().__init__()
        self.cooldown = cooldown
        self.burst_size = burst_size
        self.shot_counter = 0
        self.shot_spacing = 50
        self.next_shot = self.get_next_shot_time()

    def set_level_improvement(self, improvement_factor: float) -> None:
        self.cooldown = self.cooldown // improvement_factor
        self.improvement_factor = improvement_factor

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

    def get_points(self) -> float:
        """Returns the value of the behavior. Used to determine difficulty of host enemy

        Returns:
            float: value of movement behavior
        """

        return self.improvement_factor * self.burst_size

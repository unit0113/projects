import pygame
from .behavior import Behavior


class ForwardBehavior(Behavior):
    def __init__(self, speed: int) -> None:
        self._can_fire = True
        self.speed = speed
        self.vel_vector = pygame.Vector2(0, 1)

    def update(self, dt: float) -> None:
        self.movement = dt * self.vel_vector * self.speed

    def get_movement(self) -> pygame.Vector2:
        return self.movement

import pygame
from .behavior import Behavior


class StallBehavior(Behavior):
    def __init__(self, speed: int) -> None:
        self._can_fire = False
        self.speed = speed
        self.vel_vector = pygame.Vector2(0, 1)
        self.timer = 0
        self.stage = 0

    def update(self, dt: float) -> None:
        self.movement = dt * self.vel_vector * self.speed
        self.timer += dt

        # Come onto screen and stop at top
        if self.stage == 0 and self.timer > 2:
            self.vel_vector.y = 0
            self.stage = 1
            self.timer = 0
            self._can_fire = True
        # Move to right
        elif self.stage == 1 and self.timer > 1:
            self.vel_vector.x = 1
            self.timer = 0
            self.stage = 2
        # Stop
        elif self.stage == 2 and self.timer > 2:
            self.vel_vector.x = 0
            self.timer = 0
            self.stage = 3
        # Move to left
        elif self.stage == 3 and self.timer > 1:
            self.vel_vector.x = -0.5  # Unsure why this needs to be 0.5
            self.timer = 0
            self.stage = 4
        # Stop
        elif self.stage == 4 and self.timer > 2:
            self.vel_vector.x = 0
            self.timer = 0
            self.stage = 5
        # Move down
        elif self.stage == 5 and self.timer > 1:
            self.vel_vector.y = 2
            self.timer = 0
            self.stage = 6

    def get_movement(self) -> pygame.Vector2:
        return self.movement

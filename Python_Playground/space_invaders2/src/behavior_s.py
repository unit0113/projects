import pygame
from .behavior import Behavior


class SBehavior(Behavior):
    def __init__(self, speed: int, jerk: float = 1, direction: str = "r") -> None:
        self._can_fire = False
        self.speed = speed
        self.direction = 1 if "r" in direction.lower() else -1
        self.vel_vector = pygame.Vector2(2 * self.direction * jerk, 1)
        self.accel_vector = pygame.Vector2(0, 0)
        self.timer = 0
        self.state = 0
        self.accel_magnitude = 0.02 * jerk

    def update(self, dt: float) -> None:
        self.movement = (
            dt * self.vel_vector * self.speed
            + 0.5 * self.speed * self.accel_vector * dt * dt
        )
        self.vel_vector += self.accel_vector
        self.timer += dt

        # Come onto screen start to slow
        if self.state == 0 and self.timer > 1:
            self.accel_vector.x = -self.direction * self.accel_magnitude
            self.timer = 0
            self.state = 1
            self._can_fire = True
        # Stop lateral movement
        elif self.state == 1 and self.timer > 2.5:
            self.accel_vector.x = 0
            self.timer = 0
            self.state = 2
        # Wait and accel to right
        elif self.state == 2 and self.timer > 1.5:
            self.accel_vector.x = self.direction * self.accel_magnitude
            self.timer = 0
            self.state = 3
        elif self.state == 3 and self.timer > 2.5:
            self.accel_vector.x = 0
            self.timer = 0
            self.state = 4

    def get_movement(self) -> pygame.Vector2:
        return self.movement

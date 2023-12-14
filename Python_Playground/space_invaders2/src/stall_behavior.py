import pygame
from .behavior import Behavior


class StallBehavior(Behavior):
    def __init__(self, speed: int, jerk: float = 1, direction: str = "r") -> None:
        self._can_fire = False
        self.speed = speed
        self.vel_vector = pygame.Vector2(0, 1 * jerk)
        self.accel_vector = pygame.Vector2(0, 0)
        self.timer = 0
        self.state = 0
        self.accel_magnitude = 0.02 * jerk
        self.direction = 1 if "r" in direction.lower() else -1
        self.jerk = jerk

    def update(self, dt: float) -> None:
        self.movement = (
            dt * self.vel_vector * self.speed
            + 0.5 * self.speed * self.accel_vector * dt * dt
        )
        self.vel_vector += self.accel_vector
        self.timer += dt

        # Come onto screen start to slow
        if self.state == 0 and self.timer > 1 / self.jerk:
            self.accel_vector.y = -self.accel_magnitude
            self.state = 1
        # Come to stop
        elif self.state == 1 and self.vel_vector.y <= 0:
            self.vel_vector.y = 0
            self.accel_vector.y = 0
            self.timer = 0
            self.state = 2
            self._can_fire = True
        # Wait and accel to right
        elif self.state == 2 and self.timer > 1:
            self.accel_vector.x = self.direction * self.accel_magnitude
            self.timer = 0
            self.state = 3
        # Drift to right
        elif self.state == 3 and self.timer > 1 / self.jerk:
            self.accel_vector.x = 0
            self.timer = 0
            self.state = 4
        # Accel to left
        elif self.state == 4 and self.timer > 1:
            self.accel_vector.x = self.direction * -self.accel_magnitude
            self.timer = 0
            self.state = 5
        # Drift to left
        elif self.state == 5 and self.timer > 2 / self.jerk:
            self.accel_vector.x = 0
            self.timer = 0
            self.state = 6
        # Decelerate
        elif self.state == 6 and self.timer > 1:
            self.accel_vector.x = self.direction * self.accel_magnitude
            self.timer = 0
            self.state = 7
        # Stop
        elif self.state == 7 and (
            (self.direction == 1 and self.vel_vector.x >= 0)
            or (self.direction == -1 and self.vel_vector.x <= 0)
        ):
            self.vel_vector.x = 0
            self.accel_vector.x = 0
            self.timer = 0
            self.state = 8
        # Accel down
        elif self.state == 8 and self.timer > 1:
            self.accel_vector.y = 2 * self.accel_magnitude
            self.timer = 0
            self.state = 9
        # Coast down
        elif self.state == 9 and self.timer > 2 / self.jerk:
            self.accel_vector.y = 0
            self.timer = 0
            self.state = 10

    def get_movement(self) -> pygame.Vector2:
        return self.movement

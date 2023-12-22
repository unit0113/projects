import pygame
import random

from .behavior import Behavior
from .settings import BASE_SPEED, WIDTH


class ZigZagBehavior(Behavior):
    valid_start_locations = {
        "r": {"start_x": 25, "end_x": WIDTH // 2 - 75, "start_y": -50, "end_y": -50},
        "l": {
            "start_x": WIDTH // 2 - 75,
            "end_x": WIDTH - 75,
            "start_y": -50,
            "end_y": -50,
        },
    }

    def __init__(self, speed: float) -> None:
        self._can_fire = False
        self.speed = speed
        self.accel_vector = pygame.Vector2(0, 0)
        self.timer = 0
        self.state = 0

    def set_starting_values(self, jerk: float = 0, direction: str = "r") -> None:
        """Set starting behavior values from enemy factory

        Args:
            jerk (float, optional): derivative of acceleration. Defaults to 0.
            direction (str, optional): direction of movement. Defaults to "r".
        """

        self.vel_vector = pygame.Vector2(0, 1 * jerk)
        self.accel_magnitude = 0.02 * jerk
        self.direction = 1 if "r" in direction.lower() else -1
        self.jerk = jerk
        self.wait_time = 1 / self.jerk + random.random()

    def update(self, dt: float) -> None:
        """Calculate movement for this frame

        Args:
            dt (float): time since last frame
        """

        self.movement = (
            dt * self.vel_vector * self.speed
            + 0.5 * self.speed * self.accel_vector * dt * dt
        )
        self.vel_vector += self.accel_vector
        self.timer += dt

        # Come onto screen start to slow
        if self.state == 0 and self.timer > self.wait_time / self.jerk:
            self.accel_vector.y = -self.accel_magnitude
            self.state = 1
        # Come to stop
        elif self.state == 1 and self.vel_vector.y <= 0:
            self.vel_vector.y = 0
            self.accel_vector.y = 0
            self.timer = 0
            self.state = 2
            self._can_fire = True
        # Wait and accel to first direction
        elif self.state == 2 and self.timer > 1:
            self.accel_vector.x = self.direction * self.accel_magnitude
            self.timer = 0
            self.state = 3
        # Drift
        elif self.state == 3 and self.timer > BASE_SPEED / (self.jerk * self.speed):
            self.accel_vector.x = 0
            self.timer = 0
            self.state = 4
        # Decelerate
        elif self.state == 4 and self.timer > 1:
            self.accel_vector.x = -self.direction * self.accel_magnitude
            self.timer = 0
            self.state = 5
        # Wait
        elif self.state == 5 and self.vel_vector.x * -self.direction > 0:
            self.accel_vector.x = 0
            self.vel_vector.x = 0
            self.timer = 0
            self.state = 6
        # Accel to opposite direction
        elif self.state == 6 and self.timer > 1:
            self.accel_vector.x = self.direction * -self.accel_magnitude
            self.timer = 0
            self.state = 7
        # Drift
        elif self.state == 7 and self.timer > BASE_SPEED / (self.jerk * self.speed):
            self.accel_vector.x = 0
            self.timer = 0
            self.state = 8
        # Decelerate
        elif self.state == 8 and self.timer > 1:
            self.accel_vector.x = self.direction * self.accel_magnitude
            self.timer = 0
            self.state = 9
        # Wait and repeat
        elif self.state == 9 and self.vel_vector.x * self.direction > 0:
            self.accel_vector.x = 0
            self.vel_vector.x = 0
            self.timer = 0
            self.state = 2

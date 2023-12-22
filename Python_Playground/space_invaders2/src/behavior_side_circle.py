import pygame
import random

from .behavior import Behavior
from .settings import BASE_SPEED, WIDTH, HEIGHT


class SideCircleBehavior(Behavior):
    valid_start_locations = {
        "r": {"start_x": -50, "end_x": -50, "start_y": 40, "end_y": HEIGHT // 3},
        "l": {"start_x": WIDTH, "end_x": WIDTH, "start_y": 40, "end_y": HEIGHT // 3},
    }

    def __init__(self, speed: float) -> None:
        self._can_fire = False
        self.speed = speed
        self.accel_vector = pygame.Vector2(0, 0)
        self.timer = 0
        self.state = 0
        self.wait_time = (1.5 + random.random() * 3) / (self.speed / BASE_SPEED)

    def set_starting_values(self, jerk: float = 0, direction: str = "r") -> None:
        """Set starting behavior values from enemy factory

        Args:
            jerk (float, optional): derivative of acceleration. Defaults to 0.
            direction (str, optional): direction of movement. Defaults to "r".
        """

        self.direction = 1 if "r" in direction.lower() else -1
        self.vel_vector = pygame.Vector2(self.direction, 0)
        self.accel_magnitude = 0.01 * jerk * (self.speed / BASE_SPEED)
        self.circle_function = (
            self._move_perpendicular_counter_clockwise
            if "r" in direction.lower()
            else self._move_perpendicular_clockwise
        )
        self.jerk = jerk

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

        # Move onto screen
        if self.state == 0 and self.timer > self.wait_time:
            self.state = 1
            self.timer = 0
        # Orbit
        elif self.state == 1:
            self.circle_function()
            self._can_fire = True
            if self.timer > 4 and self._aligned():
                self.vel_vector.x = self.direction
                self.vel_vector.y = 0
                self.accel_vector *= 0
                self.state = 2

    def get_movement(self) -> pygame.Vector2:
        return self.movement

    def _move_perpendicular_clockwise(self) -> None:
        self.accel_vector.x = self.vel_vector.y * self.accel_magnitude
        self.accel_vector.y = -self.vel_vector.x * self.accel_magnitude

    def _move_perpendicular_counter_clockwise(self) -> None:
        self.accel_vector.x = -self.vel_vector.y * self.accel_magnitude
        self.accel_vector.y = self.vel_vector.x * self.accel_magnitude

    def _aligned(self) -> bool:
        return 0.99 < self.direction * self.vel_vector.x < 1.01

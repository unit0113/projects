import pygame
import random

from .behavior import Behavior
from .settings import ENEMY_BASE_SPEED, WIDTH


class StallBehavior(Behavior):
    valid_start_locations = {
        "r": {"start_x": 25, "end_x": WIDTH // 2 - 50, "start_y": -50, "end_y": -50},
        "l": {
            "start_x": WIDTH // 2 - 75,
            "end_x": WIDTH - 50,
            "start_y": -50,
            "end_y": -50,
        },
    }

    group_data = {
        "max_group_size": 4,
        "spawn_timing": "simultaneous",
        "starting_positions": [
            {
                "start_x": 25,
                "end_x": 150,
                "start_y": -50,
                "end_y": -50,
                "direction": "r",
            },
            {
                "start_x": 200,
                "end_x": WIDTH // 2 - 50,
                "start_y": -50,
                "end_y": -50,
                "direction": "r",
            },
            {
                "start_x": WIDTH // 2 - 75,
                "end_x": WIDTH // 2 + 75,
                "start_y": -50,
                "end_y": -50,
                "direction": "l",
            },
            {
                "start_x": WIDTH // 2 + 125,
                "end_x": WIDTH - 50,
                "start_y": -50,
                "end_y": -50,
                "direction": "l",
            },
        ],
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
        self.wait_time = 1 / self.jerk + 1.5 * random.random()

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
        elif self.state == 2 and self.timer > self.wait_time:
            self.accel_vector.x = self.direction * self.accel_magnitude
            self.timer = 0
            self.state = 3
        # Drift
        elif self.state == 3 and self.timer > ENEMY_BASE_SPEED / (
            self.jerk * self.speed
        ):
            self.accel_vector.x = 0
            self.timer = 0
            self.state = 4
        # Accel to opposite direction
        elif self.state == 4 and self.timer > 1:
            self.accel_vector.x = self.direction * -self.accel_magnitude
            self.timer = 0
            self.state = 5
        # Drift
        elif self.state == 5 and self.timer > 2 * ENEMY_BASE_SPEED / (
            self.jerk * self.speed
        ):
            self.accel_vector.x = 0
            self.timer = 0
            self.state = 6
        # Decelerate
        elif self.state == 6 and self.timer > 1:
            self.accel_vector.x = self.direction * self.accel_magnitude
            self.timer = 0
            self.state = 7
        # Stop
        elif self.state == 7 and self.vel_vector.x * -self.direction > 0:
            self.vel_vector.x = 0
            self.accel_vector.x = 0
            self.timer = 0
            self.state = 8
        # Accel down
        elif self.state == 8 and self.timer > self.wait_time:
            self.accel_vector.y = 2 * self.accel_magnitude
            self.timer = 0
            self.state = 9
        # Coast down
        elif self.state == 9 and self.timer > 2 / self.jerk:
            self.accel_vector.y = 0
            self.timer = 0
            self.state = 10

    def get_points(self) -> float:
        """Returns the value of the behavior. Used to determine difficulty of host enemy

        Returns:
            float: value of movement behavior
        """

        return self.jerk * self.speed / ENEMY_BASE_SPEED

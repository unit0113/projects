import pygame

from .behavior import Behavior
from .settings import ENEMY_BASE_SPEED
from .settings import WIDTH


class SBehavior(Behavior):
    valid_start_locations = {
        "r": {"start_x": -50, "end_x": -50, "start_y": 0, "end_y": 200},
        "l": {"start_x": WIDTH, "end_x": WIDTH, "start_y": 0, "end_y": 200},
    }

    group_data = {
        "max_group_size": 5,
        "spawn_timing": "sequential",
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

        self.direction = 1 if "r" in direction.lower() else -1
        self.vel_vector = pygame.Vector2(2 * self.direction * jerk, 1)
        self.accel_magnitude = 0.02 * jerk
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

        # Come onto screen start to slow
        if self.state == 0 and self.timer > 1 / (self.speed / ENEMY_BASE_SPEED):
            self.accel_vector.x = (
                -self.direction * self.accel_magnitude * (self.speed / ENEMY_BASE_SPEED)
            )
            self.timer = 0
            self.state = 1
            self._can_fire = True
        # Stop lateral movement
        elif self.state == 1 and self.timer > 2.5 / (self.speed / ENEMY_BASE_SPEED):
            self.accel_vector.x = 0
            self.timer = 0
            self.state = 2
        # Wait and accel to opposite direction
        elif self.state == 2 and self.timer > 1.5 / (self.speed / ENEMY_BASE_SPEED):
            self.accel_vector.x = (
                self.direction * self.accel_magnitude * (self.speed / ENEMY_BASE_SPEED)
            )
            self.timer = 0
            self.state = 3
        # Drift
        elif self.state == 3 and self.timer > 2.5 / (self.speed / ENEMY_BASE_SPEED):
            self.accel_vector.x = 0
            self.timer = 0
            self.state = 4

    def get_points(self) -> float:
        """Returns the value of the behavior. Used to determine difficulty of host enemy

        Returns:
            float: value of movement behavior
        """

        return 1.5 * self.jerk * self.speed / ENEMY_BASE_SPEED

import pygame

from .behavior import Behavior
from .settings import WIDTH


class ForwardBehavior(Behavior):
    valid_start_locations = {
        "r": {"start_x": 25, "end_x": WIDTH - 75, "start_y": -50, "end_y": -50}
    }

    def __init__(self, speed: float) -> None:
        self._can_fire = True
        self.speed = speed
        self.vel_vector = pygame.Vector2(0, 1)

    def set_starting_values(self, jerk: float = 0, direction: str = "r") -> None:
        """Set starting behavior values from enemy factory

        Args:
            jerk (float, optional): derivative of acceleration. Defaults to 0.
            direction (str, optional): direction of movement. Defaults to "r".
        """

        pass

    def update(self, dt: float) -> None:
        """Calculate movement for this frame

        Args:
            dt (float): time since last frame
        """

        self.movement = dt * self.vel_vector * self.speed

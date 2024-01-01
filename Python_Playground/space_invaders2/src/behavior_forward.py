import pygame

from .behavior import Behavior
from .settings import WIDTH, ENEMY_BASE_SPEED


class ForwardBehavior(Behavior):
    valid_start_locations = {
        "r": {"start_x": 25, "end_x": WIDTH - 50, "start_y": -50, "end_y": -50}
    }

    group_data = {
        "max_group_size": 6,
        "spawn_timing": "simultaneous",
        "starting_positions": [
            {
                "start_x": 25,
                "end_x": 125,
                "start_y": -50,
                "end_y": -50,
                "direction": "r",
            },
            {
                "start_x": 150,
                "end_x": 250,
                "start_y": -50,
                "end_y": -50,
                "direction": "r",
            },
            {
                "start_x": 275,
                "end_x": 375,
                "start_y": -50,
                "end_y": -50,
                "direction": "r",
            },
            {
                "start_x": 400,
                "end_x": 500,
                "start_y": -50,
                "end_y": -50,
                "direction": "r",
            },
            {
                "start_x": 525,
                "end_x": 625,
                "start_y": -50,
                "end_y": -50,
                "direction": "r",
            },
            {
                "start_x": 650,
                "end_x": WIDTH - 50,
                "start_y": -50,
                "end_y": -50,
                "direction": "r",
            },
        ],
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

        self.speed *= jerk

    def update(self, dt: float) -> None:
        """Calculate movement for this frame

        Args:
            dt (float): time since last frame
        """

        self.movement = dt * self.vel_vector * self.speed

    def get_points(self) -> float:
        """Returns the value of the behavior. Used to determine difficulty of host enemy

        Returns:
            float: value of movement behavior
        """

        return self.speed / ENEMY_BASE_SPEED

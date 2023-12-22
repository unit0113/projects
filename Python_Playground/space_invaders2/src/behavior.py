from abc import ABC, abstractmethod
import pygame


class Behavior(ABC):
    def __init__(self) -> None:
        self._can_fire = False

    @abstractmethod
    def update(self, dt: float) -> None:
        ...

    def can_fire(self) -> bool:
        return self._can_fire

    def get_movement(self) -> pygame.Vector2:
        """Return the movement calculated during update

        Returns:
            pygame.Vector2: 2D vector of ships movement in this frame
        """

        return self.movement

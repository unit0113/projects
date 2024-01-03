import pygame
from abc import ABC, abstractmethod


class State(ABC):
    should_exit: bool = False
    next_state = None

    def __init__(self, game) -> None:
        self.game = game

    @abstractmethod
    def update(self, dt: float, **kwargs) -> None:
        """Update game object in game loop

        Args:
            dt (float): time since last frame
        """

        ...

    @abstractmethod
    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        ...

    @abstractmethod
    def enter(self, **kwargs) -> None:
        """Actions to perform upon entering the state"""

        ...

    @abstractmethod
    def exit(self) -> None:
        """Actions to perform upon exiting the state"""

        ...

    @abstractmethod
    def process_events(self, events: list[pygame.event.Event]):
        """Handle game events

        Args:
            events (list[pygame.event.Event]): events to handle
        """

        ...

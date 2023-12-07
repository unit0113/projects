import pygame
from abc import ABC, abstractmethod


class State(ABC):
    should_exit: bool = False
    next_state = None

    @abstractmethod
    def update(self, dt: float, **kwargs) -> None:
        ...

    @abstractmethod
    def draw(self, window: pygame.Surface) -> None:
        ...

    @abstractmethod
    def enter(self, **kwargs) -> None:
        ...

    @abstractmethod
    def exit(self) -> None:
        ...

    @abstractmethod
    def process_event(self, event: pygame.event.Event):
        ...

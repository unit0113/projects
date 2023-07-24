from abc import ABC, abstractmethod


class State(ABC):
    def __init__(self, game, params):
        self.game = game

    @abstractmethod
    def update(self, dt: float, actions: float) -> None:
        pass

    @abstractmethod
    def draw(self) -> None:
        pass
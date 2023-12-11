from abc import ABC, abstractmethod


class Behavior(ABC):
    def __init__(self) -> None:
        self._can_fire = False

    @abstractmethod
    def update(self, dt: float) -> None:
        ...

    def can_fire(self) -> bool:
        return self._can_fire

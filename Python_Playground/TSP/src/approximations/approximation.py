from abc import ABC, abstractmethod


class Approximation(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self) -> tuple[float, bool]:
        pass

    @abstractmethod
    def draw(self, window) -> None:
        pass

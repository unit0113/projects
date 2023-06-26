from abc import ABC, abstractmethod


class Approximation(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self) -> tuple[list, bool]:
        pass
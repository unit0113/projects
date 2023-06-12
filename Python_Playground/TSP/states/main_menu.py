import pygame

from states.state import State
from src.colors import BLACK, MENU_PURPLE


class MainMenuState(State):
    def __init__(self, game) -> None:
        self.game = game

    def update(self, dt: float, actions: list) -> None:
        pass

    def draw(self) -> None:
        self.game.window.fill(BLACK)

        pygame.display.update()
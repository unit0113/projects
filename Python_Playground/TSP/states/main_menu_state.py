import pygame

from states.state import State
from src.colors import BLACK


class MainMenuState(State):
    def __init__(self, game) -> None:
        self.game = game
        self.timer = 0

    def update(self, dt: float, actions: list) -> None:
        pass

    def draw(self) -> None:
        self.game.window.fill(BLACK)

        # Buttons
        for button in self.game.buttons:
            button.draw(self.game.window)

        # Map
        self.game.assets_dict['map'].draw(self.game.window)

        # Title bar
        self.game.title.draw(self.game.window)

        pygame.display.update()
        
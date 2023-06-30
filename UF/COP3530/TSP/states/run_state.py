import pygame

from states.state import State
from src.colors import BLACK


class Run(State):
    def __init__(self, game, approx_fxn_name) -> None:
        self.game = game
        self.timer = 0
        self.approx_fxn = game.assets['approximations'][approx_fxn_name]
        # Identify selected button for drawing
        for button in self.game.assets['buttons']:
            if button.is_highlighted():
                self.title_button = button
                break

    def update(self, dt: float, actions: list) -> None:
        self.timer += dt

    def draw(self) -> None:
        self.game.window.fill(BLACK)

        # Selected menu button
        self.title_button.draw()

        # Map
        self.game.assets['map'].draw()

        pygame.display.update()
        
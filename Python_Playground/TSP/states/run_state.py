import pygame

from states.state import State
from src.colors import BLACK


class Run(State):
    def __init__(self, game, approx_fxn_index) -> None:
        self.game = game
        self.timer = 0
        self.approx_fxn = game.assets['approximations'][approx_fxn_index]

    def update(self, dt: float, actions: list) -> None:
        self.timer += dt

    def draw(self) -> None:
        self.game.window.fill(BLACK)


        pygame.display.update()
        
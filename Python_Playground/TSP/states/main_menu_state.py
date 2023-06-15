import pygame

from states.state import State
from src.colors import BLACK


class MainMenuState(State):
    def __init__(self, game) -> None:
        self.game = game
        self.timer = 0
        self.highlighted_button = None

    def update(self, dt: float, actions: list) -> None:
        # Up/down arrows highlighting menu options
        if actions[1][pygame.K_UP]:
            if self.highlighted_button == None:
                pass
                #self.highlighted_button = len(game)

        if actions[1][pygame.K_DOWN]:
            pass
            

    def draw(self) -> None:
        self.game.window.fill(BLACK)

        # Buttons
        for button in self.game.assets['buttons']:
            button.draw()

        # Map
        self.game.assets['map'].draw()

        # Title bar
        self.game.assets['title'].draw()

        pygame.display.update()
        
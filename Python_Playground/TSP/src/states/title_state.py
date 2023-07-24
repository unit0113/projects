import pygame
import pytweening as tween

from .state import State
from src.colors import BLACK
from src.settings import TITLE_TWEEN_DURATION, MAP_DELAY, MAP_TWEEN_DURATION, BUTTON_DELAY, BUTTON_FINAL_X, BUTTON_START_X, LOAD_TIME


class TitleState(State):
    def __init__(self, game, params) -> None:
        self.game = game
        self.timer = 0

        # Initiate title card tweening
        self.game.assets['title'].set_tween(tween.easeInOutQuad, 0, TITLE_TWEEN_DURATION, False, 150)

        # Initiate game button tweening
        for index, button in enumerate(self.game.assets['buttons']):
            button.set_tween(tween.easeOutSine, 1.5 + index * BUTTON_DELAY, 2, True, BUTTON_FINAL_X - BUTTON_START_X)

        # Initiate map fade in tweening
        self.game.assets['map'].set_fade_tween(tween.easeInOutQuad, MAP_DELAY, MAP_TWEEN_DURATION)

    def update(self, dt: float, actions: list) -> None:
        self.timer += dt

        # Update buttons:
        for button in self.game.assets['buttons']:
            button.update(self.timer)

        # Tween map, fade in
        self.game.assets['map'].update(self.timer)

        # Tween title card
        self.game.assets['title'].update(self.timer)

        # Transfer control to menu state
        if self.timer > LOAD_TIME:
            self.game.set_state('main_menu')

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
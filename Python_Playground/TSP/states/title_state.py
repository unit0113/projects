import pygame
import pytweening as tween

from states.state import State
from src.colors import BLACK

TITLE_HEIGHT_POS_START = -100
TITLE_HEIGHT_POS_END = 50
TITLE_OUTER_REC_SIZE = 20
TITLE_INNER_REC_SIZE = 14

TITLE_TWEEN_DURATION = 2
MAP_DELAY = 1
MAP_TWEEN_DURATION = 3
BUTTON_DELAY = 0.25

LOAD_TIME = 10


class TitleState(State):
    def __init__(self, game) -> None:
        self.game = game
        self.timer = 0

        # Initiate title card tweening
        self.game.title.set_tween(tween.easeInOutQuad, 0, TITLE_TWEEN_DURATION, False, 150)

        # Initiate game button tweening
        for index, button in enumerate(self.game.buttons):
            button.set_tween(tween.easeOutSine, 1.5 + index * BUTTON_DELAY, 2, True, 150)

        # Initiate map fade in tweening
        self.game.assets_dict['map'].set_fade_tween(tween.easeInOutQuad, MAP_DELAY, MAP_TWEEN_DURATION)

    def update(self, dt: float, actions: list) -> None:
        # Update buttons:
        for button in self.game.buttons:
            button.update(self.timer)

        # Tween map, fade in
        self.game.assets_dict['map'].update(self.timer)

        # Tween title card
        self.game.title.update(self.timer)

        # Transfer control to menu state
        if self.timer > LOAD_TIME:
            self.game.set_state('main_menu')

        self.timer += dt

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

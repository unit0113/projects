import pygame
import pytweening as tween

from states.state import State
from src.colors import BLACK
from src.settings import TITLE_TWEEN_DURATION, BUTTON_DELAY, TRANSITION_TIME, MAP_DELAY_TRANSITION, MAP_TWEEN_TRANSITION_DURATION


class Transition(State):
    def __init__(self, game, approx_fxn_index) -> None:
        self.game = game
        self.timer = 0
        self.approx_fxn_index = approx_fxn_index

        # Initiate title card tweening
        self.game.assets['title'].set_tween(tween.easeInOutQuad, 0, TITLE_TWEEN_DURATION, False, -150)

        # Initiate game button tweening
        for index, button in enumerate(self.game.assets['buttons']):
            button.set_tween(tween.easeOutSine, (len(self.game.assets['buttons']) - 1 - index) * BUTTON_DELAY, 2, True, -150)

        # Enlarge map tweening
        self.game.assets['map'].set_enlarge_tween(tween.easeInOutSine, MAP_DELAY_TRANSITION, MAP_TWEEN_TRANSITION_DURATION)

    def update(self, dt: float, actions: list) -> None:
        self.timer += dt

        # Update buttons:
        for button in self.game.assets['buttons']:
            button.update(self.timer)

        # Tween map, enlarge
        self.game.assets['map'].update(self.timer)

        # Tween title card
        self.game.assets['title'].update(self.timer)

        # Transfer control to menu state
        if self.timer > TRANSITION_TIME:
            self.game.set_state('run', self.approx_fxn_index)

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

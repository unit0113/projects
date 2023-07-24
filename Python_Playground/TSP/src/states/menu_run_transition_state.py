import pygame
import pytweening as tween

from .state import State
from src.colors import BLACK
from src.settings import TITLE_TWEEN_DURATION, BUTTON_DELAY, BUTTON_FINAL_X, BUTTON_START_X, TRANSITION_TIME, MAP_DELAY_TRANSITION, MAP_TWEEN_TRANSITION_DURATION, BUTTON_Y_LOC_RUN, BUTTON_TWEEN_DURATION


class MenuRunTransitionState(State):
    def __init__(self, game, approx_fxn_name) -> None:
        self.game = game
        self.timer = 0
        self.approx_fxn_name = approx_fxn_name

        # Initiate title card tweening
        self.game.assets['title'].set_tween(tween.easeInOutQuad, 0, TITLE_TWEEN_DURATION, False, -150)

        # Initiate game button tweening
        seen_highlighted = False
        for index, button in enumerate(self.game.assets['buttons']):
            # Move highlighted button up to act as label/title
            if button.is_highlighted():
                # Delay so no overlap, move up
                button.set_tween(tween.easeOutSine, (len(self.game.assets['buttons']) + 5) * BUTTON_DELAY, BUTTON_TWEEN_DURATION, False, -(button.start_y - BUTTON_Y_LOC_RUN))
            else:
                if seen_highlighted:
                    button.set_tween(tween.easeOutSine, (len(self.game.assets['buttons']) - index) * BUTTON_DELAY, BUTTON_TWEEN_DURATION, True, BUTTON_START_X - BUTTON_FINAL_X)
                else:
                    button.set_tween(tween.easeOutSine, (len(self.game.assets['buttons']) - 1 - index) * BUTTON_DELAY, BUTTON_TWEEN_DURATION, True, BUTTON_START_X - BUTTON_FINAL_X)

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

        # Transfer control to run state
        if self.timer > TRANSITION_TIME:
            self.game.set_state('run', self.approx_fxn_name)

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
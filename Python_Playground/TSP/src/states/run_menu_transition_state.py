import pygame
import pytweening as tween

from .state import State
from src.colors import BLACK
from src.settings import MAP_TWEEN_TRANSITION_DURATION, BUTTON_TWEEN_DURATION, BUTTON_Y_LOC_RUN, BUTTON_DELAY, BUTTON_FINAL_X, BUTTON_START_X, TITLE_TWEEN_DURATION, TRANSITION_TIME


class RunMenuTransitionState(State):
    def __init__(self, game, params) -> None:
        self.game = game
        self.timer = 0

        # Return map to original size
        self.game.assets['map'].set_shrink_tween(tween.easeInOutSine, 0, MAP_TWEEN_TRANSITION_DURATION)

        # Initiate title card tweening
        self.game.assets['title'].set_tween(tween.easeInOutQuad, MAP_TWEEN_TRANSITION_DURATION, TITLE_TWEEN_DURATION, False, 150)

        # Initiate game button tweening
        seen_highlighted = False
        for index, button in enumerate(self.game.assets['buttons']):
            # Move highlighted button up to act as label/title
            if button.is_highlighted():
                # Move to original position
                button.set_tween(tween.easeOutSine, 0, BUTTON_TWEEN_DURATION, False, -button.range)  #button.start_y - button.range - BUTTON_Y_LOC_RUN
                button.dehighlight()
                seen_highlighted = True
            else:
                # Wait for highlighted button to return
                if seen_highlighted:
                    button.set_tween(tween.easeOutSine, BUTTON_TWEEN_DURATION + (len(self.game.assets['buttons']) - index) * BUTTON_DELAY, 2, True, BUTTON_FINAL_X - BUTTON_START_X)
                else:
                    button.set_tween(tween.easeOutSine, BUTTON_TWEEN_DURATION + (len(self.game.assets['buttons']) - 1 - index) * BUTTON_DELAY, 2, True, BUTTON_FINAL_X - BUTTON_START_X)

    def update(self, dt: float, actions: list) -> None:
        self.timer += dt

        # Update buttons:
        for button in self.game.assets['buttons']:
            button.update(self.timer)

        # Tween map, shrink
        self.game.assets['map'].update(self.timer)

        # Tween title card
        self.game.assets['title'].update(self.timer)

        # Transfer control to menu state
        if self.timer > TRANSITION_TIME + 1:
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
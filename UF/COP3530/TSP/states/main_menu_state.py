import pygame

from states.state import State
from src.colors import BLACK
from src.settings import KEY_PRESS_DELAY


class MainMenuState(State):
    def __init__(self, game, params) -> None:
        self.game = game
        self.timer = 0
        self.highlighted_button = None
        self.last_key_press = 0

    def update(self, dt: float, actions: list) -> None:
        self.timer += dt

        # Up/down arrows highlighting menu options
        if actions[1][pygame.K_UP] and self.timer - self.last_key_press > KEY_PRESS_DELAY:
            # Select last if no button is highlighted
            if self.highlighted_button == None:
                self.highlighted_button = len(self.game.assets['buttons']) - 1
            else:
                # Dehighlight old, loop around if necessary
                self.game.assets['buttons'][self.highlighted_button].dehighlight()
                self.highlighted_button -= 1
                if self.highlighted_button < 0:
                    self.highlighted_button = len(self.game.assets['buttons']) - 1
            # Highlight new button and store when the last change was made
            self.game.assets['buttons'][self.highlighted_button].highlight()
            self.last_key_press = self.timer

        # Down arrow
        if actions[1][pygame.K_DOWN] and self.timer - self.last_key_press > KEY_PRESS_DELAY:
            # Select first if no button is highlighted
            if self.highlighted_button == None:
                self.highlighted_button = 0
            else:
                # Dehighlight old, loop around if necessary
                self.game.assets['buttons'][self.highlighted_button].dehighlight()
                self.highlighted_button += 1
                if self.highlighted_button > len(self.game.assets['buttons']) - 1:
                    self.highlighted_button = 0
            # Highlight new button and store when the last change was made
            self.game.assets['buttons'][self.highlighted_button].highlight()
            self.last_key_press = self.timer

        # If mouse selection
        for event in actions[0]:
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                for index, button in enumerate(self.game.assets['buttons']):
                    if button.is_clicked(pos):
                        if self.highlighted_button != None:
                            self.game.assets['buttons'][self.highlighted_button].dehighlight()
                        self.highlighted_button = index
                        self.game.assets['buttons'][self.highlighted_button].highlight()
                        break

        # If enter is pressed
        if actions[1][pygame.K_RETURN]:
            # If quit
            if self.highlighted_button == len(self.game.assets['buttons']) - 1:
                pygame.quit()
                quit()
            elif self.highlighted_button != None:
                self.game.set_state('transition', self.highlighted_button)
            

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
        
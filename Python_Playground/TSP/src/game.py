import pygame
import time

from src.settings import HEIGHT, WIDTH, FPS
from states.title_state import TitleState
from states.main_menu import MainMenuState

from src.button import Button
from src.genetic_approximation import GeneticApproximation


BUTTON_SPACING = 15


class Game:
    def __init__(self) -> None:
        # Initilize pygame modules
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("TSP Approximation")
        pygame.font.init()

        # Assets
        self._load_assets()
        self._create_buttons()

        # Game states
        self.state_dict = {'title': TitleState, 'main_menu': MainMenuState}
        self.state = self.state_dict['title'](self)

        # Approximation functions
        self.approx_functions_dict = {'genetic': GeneticApproximation}

        # Timing members
        self.run = True
        self.clock = pygame.time.Clock()
        self.prev_time = time.time()

    def _load_assets(self) -> None:
        self.assets_dict = {}
        self.assets_dict['map'] = pygame.image.load(r'assets\florida.jpg').convert_alpha()
        # Start map as transparent
        self.assets_dict['map'].set_alpha(0)

    def _create_buttons(self):
        self.buttons = []
        self.buttons.append(Button(-150, 500, "Genetic"))
        self.buttons.append(Button(-150, 500 + self.buttons[-1].rect_outer.height + BUTTON_SPACING, "Quit"))

    def set_state(self, new_state: str) -> None:
        self.state = self.state_dict[new_state](self)

    def game_loop(self) -> None:
        self.clock.tick(FPS)

        # Exit gracefully if close button is pressed
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_q]:
            pygame.quit()
            quit()

        if keys[pygame.K_r]:
            self.__init__()

        # For FPS independence
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        # Run state functions
        self.state.update(dt, [events, keys])
        self.state.draw()

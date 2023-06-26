import pygame
import time

from src.settings import HEIGHT, WIDTH, FPS, MAP_X, MAP_Y, BUTTON_SPACING, BUTTON_START_X, BUTTON_START_Y
from src.colors import BLACK, MENU_PURPLE

from states.title_state import TitleState
from states.main_menu_state import MainMenuState
from states.transition_state import Transition
from states.run_state import Run

from src.button import Button
from src.image import Image
from src.approximations.genetic_approximation import GeneticApproximation


class Game:
    def __init__(self) -> None:
        # Initilize pygame modules
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("TSP Approximation")
        pygame.font.init()

        # Assets
        self._load_assets()

        # Game states
        self.state_dict = {'title': TitleState, 'main_menu': MainMenuState, 'transition': Transition, 'run': Run}
        self.state = self.state_dict['title'](self, None)

        # Approximation functions
        self.approx_functions_dict = {'genetic': GeneticApproximation}

        # Timing members
        self.run = True
        self.clock = pygame.time.Clock()
        self.prev_time = time.time()

    def _load_assets(self) -> None:
        self.assets = {}
        
        # Map
        self.assets['map'] = Image(self.window, MAP_X, MAP_Y)

        # Title Card
        title_font = pygame.font.SysFont('verdana', 40, bold=True)
        title_text = title_font.render('Traveling Salesman Problem Approximation', 1, MENU_PURPLE) 
        self.assets['title'] = Button(self.window, WIDTH, -100, "Traveling Salesman Problem Approximation", title_text.get_width(), 40)

        # Menu buttons
        buttons = []
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y, "Greedy"))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + buttons[-1].rect_outer.height + BUTTON_SPACING, "2-Opt"))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 2 * (buttons[-1].rect_outer.height + BUTTON_SPACING), "3-Opt"))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 3 * (buttons[-1].rect_outer.height + BUTTON_SPACING), "Genetic"))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 4 * (buttons[-1].rect_outer.height + BUTTON_SPACING), "Quit"))
        self.assets['buttons'] = buttons

        # Approximation functions
        fxn = []
        fxn.append(GeneticApproximation)
        self.assets['approximations'] = fxn

    def set_state(self, new_state: str, params=None) -> None:
        self.state = self.state_dict[new_state](self, params)

    def game_loop(self) -> None:
        self.clock.tick(FPS)

        # Exit gracefully if close button is pressed
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()

        # Quit if escape is pressed
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            quit()

        # Reset if r is pressed
        if keys[pygame.K_r]:
            self.__init__()

        # For FPS independence
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        # Run state functions
        self.state.update(dt, [events, keys])
        self.state.draw()

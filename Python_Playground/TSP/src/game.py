import pygame
import time
import random

from src.settings import HEIGHT, WIDTH, FPS, MAP_X, MAP_Y, BUTTON_SPACING, BUTTON_START_X, BUTTON_START_Y
from src.colors import MENU_PURPLE

from src.states import TitleState, MainMenuState, MenuRunTransitionState, RunState, RunMenuTransitionState
from src.approximations import NearestNeighbor, Greedy, Opt2, Opt3, DivideAndConquer, NearestInsertion, FurthestInsertion, Christofides, TabuSearch, Genetic, SimmulatedAnnealing, AntColonyOptimization, BeeColonyOptimization, BlackHoleOptimization, ParticleSwarmOptimization, BruteForce
from src import Button, Image, City


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
        self.state_dict = {'title': TitleState, 'main_menu': MainMenuState, 'transition_to_run': MenuRunTransitionState, 'run': RunState, 'transition_to_menu': RunMenuTransitionState}
        self.state = self.state_dict['title'](self, None)

        # Timing members
        self.run = True
        self.clock = pygame.time.Clock()
        self.prev_time = time.time()

    def _load_assets(self) -> None:
        """ Loads primary game assests into game object
        """

        self.assets = {}
        
        # Map
        self.assets['map'] = Image(self.window, MAP_X, MAP_Y)

        # Title Card
        title_font = pygame.font.SysFont('verdana', 40, bold=True)
        title_text = title_font.render('Traveling Salesman Problem Approximation', 1, MENU_PURPLE) 
        self.assets['title'] = Button(self.window, WIDTH, -100, "Traveling Salesman Problem Approximation", title_text.get_width(), 40)

        # Approximation functions, requires Python v3.7 or later for ordered dicts
        self.assets['approximations'] = {'Nearest Neighbor': NearestNeighbor,
                                      'Greedy Heuristic': Greedy,
                                      'Greedy + 2-Opt': Opt2,
                                      'Greedy + 3-Opt': Opt3,
                                      'Divide and Conquer': DivideAndConquer,
                                      'Nearest Insertion': NearestInsertion,
                                      'Furthest Insertion': FurthestInsertion,
                                      'Cristofides': Christofides,
                                      'Tabu Search': TabuSearch,
                                      'Genetic': Genetic,
                                      'Simulated Annealing': SimmulatedAnnealing,
                                      'Ant Colony Optimization': AntColonyOptimization,
                                      'Bee Colony Optimization': BeeColonyOptimization,
                                      'Black Hole Optimization': BlackHoleOptimization,
                                      'Brute Force': BruteForce}

        # Menu buttons
        buttons = []
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y, 'Nearest Neighbor'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + buttons[-1].rect_outer.height + BUTTON_SPACING, 'Greedy Heuristic'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 2 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Greedy + 2-Opt'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 3 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Greedy + 3-Opt'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 4 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Divide and Conquer'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 5 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Nearest Insertion'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 6 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Furthest Insertion'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 7 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Cristofides'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 8 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Tabu Search'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 9 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Genetic'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 10 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Simulated Annealing'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 11 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Ant Colony Optimization'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 12 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Bee Colony Optimization'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 13 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Black Hole Optimization'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 14 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Brute Force'))
        buttons.append(Button(self.window, BUTTON_START_X, BUTTON_START_Y + 15 * (buttons[-1].rect_outer.height + BUTTON_SPACING), 'Quit'))
        self.assets['buttons'] = buttons

        # Load cities from file
        self.assets['cities'] = self.get_cities(self.window)

    def get_cities(self, window, num_cities: int=200) -> list[City]:
        """Loads specified number of cities from file and shuffles

        Args:
            window (pygame.surface.Surface): pygame window object for City init
            num_cities (int): Number of cities to load

        Returns:
            list[City]: randomized list of cities
        """

        with open(r'assets/cities.csv', 'r') as file:
            cities = [City(window, *params.strip().split(',')) for params in file.readlines()[1:num_cities + 1]]

        random.shuffle(cities)
        return cities

    def calculate_city_XY(self):
        """ Set X, Y values of city based on size and position of map object
        """
        image_start_x = self.assets['map'].x
        image_start_y = self.assets['map'].y
        image_height = self.assets['map'].start_height
        image_width = self.assets['map'].start_width

        for city in self.assets['cities']:
            city.calculate_XY(image_start_x, image_start_y, image_height, image_width)

    def set_state(self, new_state: str, params=None) -> None:
        """ Advances game state

        Args:
            new_state (str): Name of next game state as specified is self.state_dict
            params (_type_, optional): Any required initializer paramaters for selected game state. Defaults to None.
        """

        self.state = self.state_dict[new_state](self, params)

    def game_loop(self) -> None:
        """ Main game loop
            Allows user to quit by pressing the close button or by pressing the Esc key
            Allows user to restart by pressing the r key
            Calls the current game states update and draw functions
        """

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
        
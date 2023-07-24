#https://www.pygame.org/wiki/MatplotlibPygame

import pygame
import matplotlib
matplotlib.use('Agg')
import matplotlib.backends.backend_agg as agg
import pylab
from math import sin

from .state import State
from src.colors import BLACK, MAXIMUM_GREEN_FLOAT, MENU_PURPLE_FLOAT, MENU_PURPLE


GRAPH_WIDTH = 10
GRAPH_HEIGHT = 4
GRAPH_X = 100
GRAPH_Y = 600
CONTINUE_TEXT_SIZE = 40
CONTINUE_X = 300
CONTINUE_Y = 400


class RunState(State):
    def __init__(self, game, approx_fxn_name) -> None:
        self.game = game
        self.timer = 0
        self.approx_fxn = game.assets['approximations'][approx_fxn_name](self.game.assets['cities'])
        self.approximation_complete = False
        self.distances = []
        self.fig = pylab.figure(figsize=[GRAPH_WIDTH, GRAPH_HEIGHT], dpi=100)
        self.fig.set_facecolor(BLACK)
        self.ax = self.fig.gca()

        # Identify selected button for drawing
        for button in self.game.assets['buttons']:
            if button.is_highlighted():
                self.title_button = button
                break

        # Recalculate X, Y positions for cities
        self.game.calculate_city_XY()

        # Call an initial iteration of approximation to prime graph
        self._run_approximation()

        # Variables for continue message
        self.font = pygame.font.SysFont('verdana', CONTINUE_TEXT_SIZE, bold=True)
        self.continue_text = self.font.render("Press Space to Continue", 1, MENU_PURPLE)
        self.continue_alpha = 0
        self.continue_text.set_alpha(self.continue_alpha)
        self.tween_timer = 0

    def update(self, dt: float, actions: list) -> None:
        self.timer += dt

        # Run approximation
        if not self.approximation_complete:
            self._run_approximation()
        
        else:
            self.continue_alpha = 255 * sin(self.tween_timer)
            self.continue_text.set_alpha(self.continue_alpha)
            self.tween_timer += dt / 2

        # If complete, allow transition back to main menu
        if actions[1][pygame.K_SPACE]:
            self.game.set_state('transition_to_menu')

    def _run_approximation(self) -> None:
        """ Runs a single iteration of the approximation function and stores the results
        """

        score, self.approximation_complete = self.approx_fxn.run()
        self.distances.append(60 / score)
        self._update_graph()

    def _update_graph(self) -> None:
        """ Update data in the graph and prepare to draw
        """

        self.ax.clear()
        self.ax.plot(self.distances, color=MENU_PURPLE_FLOAT)
        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Distance')
        self.ax.set_title(f'Current Distance: {self.distances[-1]:.2f} miles', color=MAXIMUM_GREEN_FLOAT)
        self.ax.title.set_size(16)
        self.ax.set_facecolor(BLACK)
        self.ax.xaxis.label.set_color(MAXIMUM_GREEN_FLOAT)
        self.ax.yaxis.label.set_color(MAXIMUM_GREEN_FLOAT)
        self.ax.tick_params(axis='x', colors=MAXIMUM_GREEN_FLOAT)
        self.ax.tick_params(axis='y', colors=MAXIMUM_GREEN_FLOAT)
        self.ax.spines['left'].set_color(MAXIMUM_GREEN_FLOAT)
        self.ax.spines['bottom'].set_color(MAXIMUM_GREEN_FLOAT)


        self.canvas = agg.FigureCanvasAgg(self.fig)
        self.canvas.draw()
        self.raw_data = self.canvas.get_renderer().tostring_rgb()

    def draw(self) -> None:
        self.game.window.fill(BLACK)

        # Selected menu button
        self.title_button.draw()

        # Map
        self.game.assets['map'].draw()

        # Cities
        for city in self.game.assets['cities']:
            city.draw()

        # Route
        self.approx_fxn.draw(self.game.window)

        # Graph
        graph = pygame.image.fromstring(self.raw_data, self.canvas.get_width_height(), "RGB")
        self.game.window.blit(graph, (GRAPH_X, GRAPH_Y))

        # Continue message
        self.game.window.blit(self.continue_text, (CONTINUE_X, CONTINUE_Y))

        pygame.display.update()
        
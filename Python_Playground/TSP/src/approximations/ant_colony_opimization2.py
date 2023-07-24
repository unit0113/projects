# https://www.theprojectspot.com/tutorial-post/ant-colony-optimization-for-hackers/10

import math
import random

from .approximation import Approximation
from .approximation_utils import draw_route, calc_distance, calc_fitness_memo
from src.functions import randomize_route


class Edge:
    def __init__(self, a, b, heuristic, pheromone):
        self.a = a
        self.b = b
        self.heuristic = heuristic
        self.pheromone = pheromone

class Ant:
    def __init__(self, edges, alpha, beta, n_nodes):
        """
        alpha -> parameter used to control the importance of the pheromone trail
        beta  -> parameter used to control the heuristic information during selection
        """
        self.edges = edges
        self.alpha = alpha
        self.beta = beta
        self.n_nodes = n_nodes
        self.tour = self.update_tour()
        self.distance = self.calculate_distance()

    def _select_node(self):
        """
        Constructing solution
        an ant will often follow the strongest
        pheromone trail when constructing a solution.

        state -> is a point on a graph or a City

        Here, an ant would be selecting the next city depending on the distance
        to the next city, and the amount of pheromone on the path between
        the two cities.
        """
        roulette_wheel = 0
        states = [node for node in range(self.n_nodes) if node not in self.tour]
        heuristic_value = 0
        for new_state in states:
            heuristic_value += self.edges[self.tour[-1]][new_state].heuristic

        for new_state in states:
            A = math.pow(self.edges[self.tour[-1]][new_state].pheromone, self.alpha)
            B = math.pow((heuristic_value / self.edges[self.tour[-1]][new_state].heuristic), self.beta)
            roulette_wheel += A * B

        random_value = random.uniform(0, roulette_wheel)
        wheel_position = 0
        for new_state in states:
            A = math.pow(self.edges[self.tour[-1]][new_state].pheromone, self.alpha)
            B = math.pow((heuristic_value/self.edges[self.tour[-1]][new_state].heuristic), self.beta)
            wheel_position += A * B
            if wheel_position >= random_value:
                return new_state

    def update_tour(self):
        self.tour = [random.randint(0, self.n_nodes - 1)]
        while len(self.tour) < self.n_nodes:
            self.tour.append(self._select_node())
        return self.tour

    def calculate_distance(self):
        self.distance = 0
        for i in range(self.n_nodes):
            self.distance += self.edges[self.tour[i]][self.tour[(i+1)%self.n_nodes]].heuristic
        return self.distance
    

class AntColonyOptimization(Approximation):
    def __init__(self, nodes: list, num_ants: int=5, elitist_weight: float=1.0, alpha: float=1.0, beta: float=3.0,
                 rho: float=0.1, phe_deposit_weight: float=1.0, initial_pheromone_strength: float=1.0, num_iterations: int=100) -> None:
        self.num_ants = num_ants
        self.elitist_weight = elitist_weight
        self.alpha = alpha
        self.rho = rho
        self.phe_deposit_weight = phe_deposit_weight
        self.num_iterations = num_iterations
        self.current_iteration = 0
        self.n_nodes = len(nodes)
        self.nodes = nodes

        # Initialize solution space as 2D grid
        self.edges = [[None for _ in range(self.n_nodes)] for _ in range(self.n_nodes)]
        for row in range(self.n_nodes):
            for col in range(self.n_nodes):
                heuristic = calc_distance(self.nodes[row], self.nodes[col])
                self.edges[row][col] = self.edges[col][row] = Edge(row, col, heuristic, initial_pheromone_strength)

        self.ants = [Ant(self.edges, alpha, beta, self.n_nodes) for _ in range(self.num_ants)]

        self.best_tour = self.ants[0].tour
        self.best_distance = calc_fitness_memo(self.best_tour)

    def _add_pheromone(self, tour, distance):
        pheromone_to_add = self.phe_deposit_weight / distance
        for i in range(self.n_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.n_nodes]].pheromone += pheromone_to_add

    def run(self) -> tuple[float, bool]:
        """Perform a single step of the optimization

        Returns:
            tuple[float, bool]: returns the score of the top performing organism and whether the approximation is completed
        """

        for ant in self.ants:
            self._add_pheromone(ant.update_tour(), ant.calculate_distance())
            if ant.distance < self.best_distance:
                self.best_tour = ant.tour
                self.best_distance = ant.distance
        self._add_pheromone(self.best_tour, self.best_distance, self.elitist_weight)

        for x in range(self.n_nodes):
            for y in range(self.n_nodes):
                self.edges[x][y].pheromone *= (1-self.rho)

        self.current_iteration += 1
        return self.best_distance, self.current_iteration >= self.num_iterations
    
    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        draw_route(window, self.best_tour)

# https://www.theprojectspot.com/tutorial-post/ant-colony-optimization-for-hackers/10

import math
import random

from .approximation import Approximation
from .approximation_utils import draw_route, calc_distance, calc_fitness_memo


class Ant:
    def __init__(self, grid:dict[dict[float]], alpha: float, beta: float):
        self.grid = grid
        self.alpha = alpha  # Importance of existing pheromone trail
        self.beta = beta    # Controls the heuristic information during selection
        self.tour = list(self.grid.keys())
        self.n_nodes = len(self.tour)
        self.update_tour()

    def update_tour(self) -> None:
        """ Generate a new route based on pheremone trails
            Stocastically add new node to route under construction based on
            pheremone strength and distance to nodes via weighted random sampling
        """

        potential_nodes = [node for node in self.grid.keys()]
        self.tour = [random.choice(potential_nodes)]

        # Add nodes to tour
        while len(self.tour) < self.n_nodes:
            # Remove last added node from potential_nodes
            potential_nodes.remove(self.tour[-1])

            # Get total distance to possible new nodes
            heuristic_value = sum(map(lambda x: calc_distance(self.tour[-1], x), potential_nodes))

            # Build roullete wheel for weighted random sampling
            wheel_positions = [0]
            for poss_node in potential_nodes:
                A = math.pow(self.grid[self.tour[-1]][poss_node], self.alpha)
                B = math.pow((heuristic_value / (1 + calc_distance(self.tour[-1], poss_node))), self.beta)  # 1 + avoids divide by 0 error
                wheel_positions.append(wheel_positions[-1] + A * B)

            # Select new node
            random_value = random.uniform(0, wheel_positions[-1])
            del wheel_positions[0]  # Erase bogus temp value
            for node_index, new_node in enumerate(potential_nodes):
                if wheel_positions[node_index] >= random_value:
                    self.tour.append(new_node)
                    break
            else:
                self.tour.append(potential_nodes[-1])

        # Update fitness
        self.fitness = calc_fitness_memo(self.tour)

    def add_pheromones(self, pheremone_deposit_weight: float=1, heuristic: float=1.0) -> None:
        """ Adds pheremones to grid based on fitness of tour and deposit weight hyperparameter

        Args:
            pheremone_deposit_weight (float): Hyperparameter for strength of new pheremone deposits
        """

        pheromone_to_add = pheremone_deposit_weight * self.fitness
        for node_index, city in enumerate(self.tour):
            self.grid[city][self.tour[(node_index + 1) % self.n_nodes]] += heuristic * pheromone_to_add


class AntColonyOptimization(Approximation):
    def __init__(self, cities: list, num_ants: int=6, elitist_weight: float=2.0, alpha: float=1.0, beta: float=3.0,
                 rho: float=0.1, initial_pheromone_strength: float=0.0001, num_iterations: int=250) -> None:
        self.num_ants = num_ants
        self.elitist_weight = elitist_weight
        self.rho = rho  # Pheremone evaporation rate
        self.num_iterations = num_iterations
        self.current_iteration = 0
        self.n_nodes = len(cities)

        # Initialize solution space as 2D grid
        self.grid = {city: {inner_city: initial_pheromone_strength for inner_city in cities} for city in cities}

        # Zeroize self-connections
        for city in self.grid.keys():
            self.grid[city][city] = 0

        self.ants = [Ant(self.grid, alpha, beta) for _ in range(self.num_ants)]

        self.best_ant = self.ants[0]
        self.best_tour = self.best_ant.tour
        self.best_fitness = calc_fitness_memo(self.best_tour)

    def run(self) -> tuple[float, bool]:
        """Perform a single step of the optimization

        Returns:
            tuple[float, bool]: returns the score of the top performing organism and whether the approximation is completed
        """

        for ant in self.ants:
            ant.update_tour()
            ant.add_pheromones()
        
        self.best_ant = max(self.ants, key=lambda x: x.fitness)
        if self.best_ant.fitness > self.best_fitness:
            self.best_tour = self.best_ant.tour
            self.best_fitness = self.best_ant.fitness
        
        # Add extra pheremone strength on best path
        self.best_ant.add_pheromones(heuristic=self.elitist_weight)

        # Evaporate pheremones
        for row in self.grid.keys():
            for col in self.grid[row].keys():
                self.grid[row][col] *= (1-self.rho)

        self.current_iteration += 1
        return self.best_fitness, self.current_iteration >= self.num_iterations
    
    def draw(self, window) -> None:
        """ Draw ant trails

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        draw_route(window, self.best_tour)

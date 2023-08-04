#https://medium.com/@francis.allanah/travelling-salesman-problem-using-simulated-annealing-f547a71ab3c6

import random
import copy
import math

from .approximation import Approximation
from .approximation_utils import draw_route, calc_fitness_memo


BATCH_SIZE = 250
DISTANCE_CORRECTIVE_FACTOR = 5_000_000


class SimmulatedAnnealing(Approximation):
    def __init__(self, cities: list, start_temp: int=5_000, alpha: float=0.99, num_iterations: int=500) -> None:
        self.best = cities
        self.current_temp = start_temp
        self.alpha = alpha
        self.current_iteration = 0
        self.num_iterations = num_iterations
        self.mutation_functions = [self._inverse, self._insert, self._swap, self._swap_routes]

    def run(self) -> tuple[float, bool]:
        """Perform a batch of simulated annealing

        Returns:
            tuple[float, bool]: The score of the current route and whether the approximation is completed
        """

        for _ in range(BATCH_SIZE):
            self._anneal()

        self.current_iteration += 1

        return calc_fitness_memo(self.best), self.current_iteration >= self.num_iterations

    def _anneal(self) -> None:
        """ Perform a single iteration of simulated annealing
        """

        candidate = self._get_candidate(self.best)
        cost_diff = calc_fitness_memo(candidate) - calc_fitness_memo(self.best)

        # Accept if candidate is better, otherwise, accept it with a probability of e^(-cost/temp)
        if cost_diff > 0 or random.uniform(0, 1) <= math.exp(DISTANCE_CORRECTIVE_FACTOR * float(cost_diff) / float(self.current_temp)):
            self.best = candidate
            self.same_route_count = 0
                
        # Reduce current temp
        self.current_temp = self.current_temp * self.alpha
        
    def _inverse(self, state: list) -> list:
        """ Inverses the order of cities in a route between node one and node two

        Args:
            state (list): Potential TPS solution

        Returns:
            list: Potential TPS solution with inverted section
        """
    
        node_one, node_two = random.sample(range(len(state) - 1), 2)
        state[min(node_one,node_two):max(node_one,node_two)] = state[min(node_one,node_two):max(node_one,node_two)][::-1]
        
        return state
    
    def _swap(self, state: list) -> list:
        """ Swap cities at positions i and j with each other

        Args:
            state (list): Potential TPS solution

        Returns:
            list: Potential TPS solution with two positions swapped
        """

        pos_one, pos_two = random.sample(range(len(state)), 2)
        state[pos_one], state[pos_two] = state[pos_two], state[pos_one]
        
        return state
    
    def _insert(self, state: list) -> list:
        """ Insert city at node j before node i

        Args:
            state (list): Potential TPS solution

        Returns:
            list: Potential TPS solution with a city moved to a new position
        """

        node_j = random.choice(state)
        state.remove(node_j)
        index = random.randint(0, len(state) - 1)
        state.insert(index, node_j)
        
        return state
    
    def _swap_routes(self, state: list) -> list:
        """Select a subroute from a to b and insert it at another position in the route

        Args:
            state (list): Potential TPS solution

        Returns:
            list: Potential TPS solution with a subroute moved to a different location
        """

        subroute_a, subroute_b = random.sample(range(len(state)), 2)
        subroute = state[min(subroute_a, subroute_b):max(subroute_a, subroute_b)]
        del state[min(subroute_a,subroute_b):max(subroute_a, subroute_b)]
        insert_pos = random.choice(range(len(state)))
        state[insert_pos:insert_pos] = subroute

        return state
    
    def _get_candidate(self, state: list) -> list:
        """ Performs a random muation operation on the provided state

        Args:
            state (list): Potential TPS solution

        Returns:
            list: Potential TPS solution with a random mution applied
        """

        mutation_fxn1 = random.choice(self.mutation_functions)
        return mutation_fxn1(copy.copy(state))


    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        draw_route(window, self.best)

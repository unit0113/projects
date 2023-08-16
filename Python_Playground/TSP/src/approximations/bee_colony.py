import random
import copy

from . import Approximation
from . import randomize_route, calc_route_distance, draw_route


class Bee:
    def __init__(self, cities) -> None:
        self.route = cities[:]
        self.distance = calc_route_distance(self.route)
        self.mutation_functions = [self._inverse, self._insert, self._swap, self._swap_routes]
    
    def _mutate(self, route) -> list:
        """ Swap two adject cities

        Returns:
            list: New route
        """

        mutation_fxn1 = random.choice(self.mutation_functions)
        return mutation_fxn1(copy.copy(route))
    
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
    

class WorkerBee(Bee):
    def __init__(self, cities: list, same_cycle_limit: int=10) -> None:
        super().__init__(cities)
        self.num_same_cycles = 0
        self.same_cycle_limit = same_cycle_limit

    def forage(self) -> None:
        """ Explores nearby sample space. If new route is better, exploits the new route.
            If no improvement limit has been exceeded, scout new random route
        """

        new_path = self._mutate(self.route)
        
        # If new is better
        if calc_route_distance(new_path) < self.distance:
            self.route = new_path
            self.distance = calc_route_distance(self.route)
            self.num_same_cycles = 0
        
        else:
            self.num_same_cycles += 1
            # Explore new route if limit exceeded
            if self.num_same_cycles > self.same_cycle_limit:
                self.scout()
    
    def scout(self) -> None:
        """ Scouts new location by exploring random route
        """

        self.route = randomize_route(self.route)
        self.distance = calc_route_distance(self.route)
        self.num_same_cycles = 0


class OnlookerBee(Bee):
    def __init__(self, cities) -> None:
        super().__init__(cities)

    def exploit(self, route) -> None:
        """ Go to directed route and explores nearby sample space. If new route is better, store as best.
            If no improvement limit has been exceeded, scout new random route
        """

        new_path = self._mutate(route)

        # If new is better
        if calc_route_distance(new_path) < self.distance:
            self.route = new_path
            self.distance = calc_route_distance(self.route)


class BeeColonyOptimization(Approximation):
    def __init__(self, cities: list, num_worker: int=200, num_onlooker: int=1000, size_best: int=5, rescout_perc: float=0.25, num_iterations: int=500) -> None:
        self.workers = [WorkerBee(randomize_route(cities)) for _ in range(num_worker)]
        self.onlookers = [OnlookerBee(randomize_route(cities)) for _ in range(num_onlooker)]
        self.num_iterations = num_iterations
        self.curr_iterations = 0
        self.best_bees = [worker for worker in self.workers[:size_best]]
        self.size_best = size_best
        self.rescout_perc = rescout_perc

    def run(self) -> tuple[float, bool]:
        """ Perform a single step in the bee colony process.
            Forage with worker bees
            Evaluate results
            Deploy Onlookers

        Returns:
            tuple[float, bool]: returns the length of the route of the top performing bee and whether the approximation is completed
        """

        # Forage with worker bees
        for worker in self.workers:
            worker.forage()

        # Exploit with onlookers
        chunk_size = len(self.onlookers) // self.size_best
        for i, bee in enumerate(self.best_bees):
            for onlooker in self.onlookers[chunk_size*i:chunk_size*(i+1)]:
                onlooker.exploit(bee.route)

        # Store best
        self.workers.sort(key=lambda x: x.distance)
        self.onlookers.sort(key=lambda x: x.distance)
        self.best_bees = [bee for bee in sorted(self.workers[:self.size_best] + self.onlookers[:self.size_best] + self.best_bees, key=lambda x: x.distance)][:self.size_best]
        
        # Set worst workers to scout
        for worker in self.workers[int(len(self.workers) * (1-self.size_best)):]:
            worker.scout()

        self.curr_iterations += 1

        return self.best_dist, self.curr_iterations >= self.num_iterations
    
    @property
    def best(self) -> list:
        return self.best_bees[0].route
    
    @property
    def best_dist(self) -> float:
        return self.best_bees[0].distance
    
    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        draw_route(window, self.best)
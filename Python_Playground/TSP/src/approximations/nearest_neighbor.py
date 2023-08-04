import math

from .approximation import Approximation
from .approximation_utils import draw_route, calc_distance, calc_fitness_memo, randomize_route


class NearestNeighborAgent:
    def __init__(self, cities) -> None:
        self.route = [cities[0]]
        self.remaining_cities = cities[1:]
        self.fitness = calc_fitness_memo(self.route + self.remaining_cities)

    def done(self) -> bool:
        """ Tells main algorithm if there are any remaining cities to add to the route

        Returns:
            bool: Whether the algorithm is complete
        """
        return not self.remaining_cities
    
    def _add_closest(self) -> None:
        """Move the city that is closest to the last city in the route
           from the remaining list into the route list
        """
        
        # intialize values
        closest_dist = math.inf
        closest_idx = None

        # Loop through remaining, find closest
        for index, city in enumerate(self.remaining_cities):
            dist = calc_distance(self.route[-1], city)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = index

        # Move closest from remaining to route
        self.route.append(self.remaining_cities.pop(closest_idx))

        # Recalculate fitness
        self.fitness = calc_fitness_memo(self.route + self.remaining_cities)


class NearestNeighbor(Approximation):
    def __init__(self, cities: list, num_agents: int=100) -> None:
        self.agents = [NearestNeighborAgent(randomize_route(cities)) for _ in range(num_agents)]
        self.best_agent = self.agents[0]

    def run(self) -> tuple[float, bool]:
        """Add a single city to the current route

        Returns:
            tuple[float, bool]: returns the score of the current route and whether the approximation is completed
        """

        for agent in self.agents:
            agent._add_closest()

        self.best_agent = max(self.agents, key=lambda x: x.fitness)
        return self.best_agent.fitness, self.best_agent.done()
    
    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        draw_route(window, self.best_agent.route + self.best_agent.remaining_cities)

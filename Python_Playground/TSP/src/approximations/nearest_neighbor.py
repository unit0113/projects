import math

from .approximation import Approximation
from .approximation_utils import draw_route, calc_distance, calc_fitness_memo


class NearestNeighbor(Approximation):
    def __init__(self, cities: list) -> None:
        self.route = [cities[0]]
        self.remaining_cities = cities[1:]


    def run(self) -> tuple[float, bool]:
        """Add a single city to the current route

        Returns:
            tuple[float, bool]: returns the score of the current route and whether the approximation is completed
        """

        self._add_closest()
        return calc_fitness_memo(self.route + self.remaining_cities), not self.remaining_cities

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

    
    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        draw_route(window, self.route + self.remaining_cities)

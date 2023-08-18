import math

from .approximation import Approximation
from .approximation_utils import draw_route, calc_distance, calc_route_distance


class NearestInsertion(Approximation):
    def __init__(self, cities: list) -> None:
        self.route = [cities[0]]
        self.remaining_cities = cities[1:]

    def run(self) -> tuple[float, bool]:
        """Add a single city to the current route

        Returns:
            tuple[float, bool]: returns the length of the current route and whether the approximation is completed
        """

        nearest_city = self._get_nearest()
        self._add_to_route(nearest_city)
        
        return self.distance, not self.remaining_cities
    
    @property
    def distance(self) -> float:
        return calc_route_distance(self.route + self.remaining_cities)
    
    def _get_nearest(self) -> object:
        """ Get the city that is closest to any city in the current route

        Returns:
            City: City in remaining cities that is closest to a city in the current route
        """
        
        # Intialize values
        closest_dist = math.inf
        closest_idx = None

        # Loop through remaining, find closest
        for index, city in enumerate(self.remaining_cities):
            dist = min([calc_distance(city, route_city) for route_city in self.route])
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = index

        return self.remaining_cities[closest_idx]
    
    def _add_to_route(self, city: object) -> None:
        """ Insert city into current route in the position that results in the shortest route.
            Removes inserted city from remaining city

        Args:
            city (City): City to insert into route
        """

        # Remove city from remaining cities
        self.remaining_cities.remove(city)

        # Base case for route initialization
        if len(self.route) < 2:
            self.route.append(city)
            return

        # Find best insertion option
        best_route = [city] + self.route
        best_dist = calc_route_distance(best_route)
        for index in range(1, len(self.route) - 1):
            new_route = self.route[:]
            new_route.insert(index, city)
            new_dist = calc_route_distance(new_route)
            if new_dist < best_dist:
                best_dist = new_dist
                best_route = new_route

        # Store result
        self.route = best_route


    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        draw_route(window, self.route)

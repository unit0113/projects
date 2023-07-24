from itertools import permutations

from .approximation import Approximation
from .approximation_utils import draw_route, calc_distance, calc_fitness_memo


class BruteForce(Approximation):
    def __init__(self, city_list: list) -> None:
        self.start = city_list[0]
        self.routes = permutations(city_list[1:])
        self.next_route = 0
        self.minimum_route = [self.start] + list(next(self.routes))
        self.minimum_distance = 1 / calc_fitness_memo(self.minimum_route)

    def run(self) -> tuple[list, bool]:
        """Runs a single iteration of the brute force method

        Returns:
            tuple[list, bool]: Current best route, boolean on whether the function is complete
        """
        
        try:
            route = [self.start] + list(next(self.routes))
        except StopIteration:
            return 1 / self.minimum_distance, True
        
        return self._run(route)

    def _run(self, route) -> tuple[list, bool]:
        """Subfunction to run a single iteration of the brute force method

        Returns:
            tuple[list, bool]: Current best route, boolean on whether the function is complete

        Yields:
            Iterator[tuple[list, bool]]: Iterator that points to the results of a single run.
        """

        # Intialize distance to the distance between the first and last element in the route
        prev = route[0]
        current_distance = calc_distance(prev, route[-1])

        # Calc total distance for the route
        for next in route[1:]:
            current_distance += calc_distance(prev, next)
            prev = next
            # Early stopping
            if current_distance > self.minimum_distance:
                break

        # Store if best
        else:
            if current_distance < self.minimum_distance:
                self.minimum_distance = current_distance
                self.minimum_route = route

        self.next_route += 1

        return 1 / self.minimum_distance, False
    
    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """
        
        draw_route(window, self.minimum_route)

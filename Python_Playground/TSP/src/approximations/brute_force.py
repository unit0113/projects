from itertools import permutations
import sys

from .approximation import Approximation

sys.path.append('..')
from TSP.src.functions import calc_distance, calc_fitness_memo

class BruteForce(Approximation):
    def __init__(self, city_list: list) -> None:
        self.start = city_list[0]
        self.routes = self.permutation_generator(city_list)
        self.next_route = 0
        self.minimum_route = [self.start] + list(self.routes)
        self.minimum_distance = 1 / calc_fitness_memo(self.minimum_route)

    def permutation_generator(self, city_list: list) -> tuple:
        """Generator function to allow iterating through all permutations of routes.
           Not using generator results in memory error

        Args:
            city_list (list): List of cities to build a route from

        Returns:
            tuple: Route to be tested. Does not include starting city
        """

        return next(permutations(city_list[1:]))

    def run(self) -> tuple[list, bool]:
        """Runs a single iteration of the brute force method

        Returns:
            tuple[list, bool]: Current best route, boolean on whether the function is complete
        """

        brute_force_generator = self._run()
        return next(brute_force_generator)

    def _run(self) -> tuple[list, bool]:
        """Subfunction to run a single iteration of the brute force method

        Returns:
            tuple[list, bool]: Current best route, boolean on whether the function is complete

        Yields:
            Iterator[tuple[list, bool]]: Iterator that points to the results of a single run.
        """

        # Convert route from permutation generator into list, include starting city
        route = [self.start] + list(self.routes)
        prev = route[0]

        # Intialize distance to the distance between the first and last element in the route
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

        yield self.minimum_route, self.next_route >= len(self.routes)
    
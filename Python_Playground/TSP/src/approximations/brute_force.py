from itertools import permutations
import sys

from .approximation import Approximation

sys.path.append('..')
from TSP.src.functions import calc_distance

class BruteForce(Approximation):
    def __init__(self, city_list) -> None:
        self.cities = city_list

    def run(self) -> tuple[list, bool]:
        brute_force_generator = self._run()
        return next(brute_force_generator)

    def _run(self):
        minimum_distance = float('inf')
        minimum_route = self.cities

        for route in permutations(self.cities[1:]):
            current_distance = 0

            prev = self.cities[0]
            for next in route:
                current_distance += calc_distance(prev, next)
                prev = next
                if current_distance > minimum_distance:
                    break

            else:
                if current_distance < minimum_distance:
                    minimum_distance = current_distance
                    minimum_route = [self.cities[0]] + list(route)
            
            yield minimum_route, False

        yield minimum_route, True
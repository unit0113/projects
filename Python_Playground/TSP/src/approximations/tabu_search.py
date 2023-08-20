# https://archive.ph/Wawlb

from .approximation import Approximation
from .furthest_insertion import FurthestInsertion
from .approximation_utils import draw_route, calc_route_distance


class TabuSearch(Approximation):
    def __init__(self, cities: list, tabu_iterations: int=25) -> None:
        self.initial_approx = FurthestInsertion(cities)
        self.best = cities
        self.initial_done = False
        self.tabu_iterations = tabu_iterations
        self.curr_tabu_iterations = 0
        self.tabu_list = []

    def run(self) -> tuple[float, bool]:
        """ Perform greedy heuristic to seed initial route, then iterate through route to find any possible improvements

        Returns:
            tuple[float, bool]: returns the length of the current route and whether the approximation is completed
        """

        if not self.initial_done:
            _, self.initial_done = self.initial_approx.run()
            self.best = self.initial_approx.route + self.initial_approx.remaining_cities
            if self.initial_done:
                self.best_candidate = self.best

        else:
            self._tabu_search()
            self.curr_tabu_iterations += 1

        return calc_route_distance(self.best), self.initial_done and self.curr_tabu_iterations >= self.tabu_iterations
    
    def _tabu_search(self, size: int=10) -> None:
        """ Compares best swap neighbor to current best solution

        Args:
            size (int, optional): Max size of tabu list. Defaults to 10.
        """

        # Find best candidate
        self.best_candidate = self._best_neighbor()
        best_dist = calc_route_distance(self.best_candidate)

        # Replace best as requried
        if best_dist < calc_route_distance(self.best):
            self.best = self.best_candidate
            
        # Remove from tabu list if enough iterations have passed
        if len(self.tabu_list) >= size:
            self.tabu_list.pop(0)
    
    def _best_neighbor(self) -> list:
        """ Finds the swap that creates the route with the lowest total distance

        Returns:
            list: best new route found
        """

        # Add current best to tabu list
        self.tabu_list.append(self.best_candidate)
        neighborhood = []

        # Loop through possible swaps
        for index1, city1 in enumerate(self.best_candidate):
            for index2, city2 in enumerate(self.best_candidate[index1+1:]):
                index2 += 1 + index1

                # Make swap
                tmp = self.best_candidate[:]
                tmp[index1], tmp[index2] = city2, city1

                if tmp not in self.tabu_list:
                    neighborhood.append(tmp)

        return min(neighborhood, key=calc_route_distance)
    
    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        if not self.initial_done:
            self.initial_approx.draw(window)
        else:
            draw_route(window, self.best)

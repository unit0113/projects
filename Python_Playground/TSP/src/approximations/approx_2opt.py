# https://colab.research.google.com/github/norvig/pytudes/blob/main/ipynb/TSP.ipynb#scrollTo=PBZUC3S6aUet

from .approximation import Approximation
from .greedy import Greedy
from .approximation_utils import draw_edges, draw_route, calc_distance, calc_fitness_memo, randomize_route


class Opt2(Approximation):
    def __init__(self, cities: list) -> None:
        self.greedy_approx = Greedy(cities)
        self.best = cities
        self.greedy_done = False
        self.opt_done = False
        self.subsegments = self.get_subsegments()

    def run(self) -> tuple[float, bool]:
        """ Perform greedy heuristic to seed initial route, then iterate through route to find any possible improvements

        Returns:
            tuple[float, bool]: returns the score of the current route and whether the approximation is completed
        """

        if not self.greedy_done:
            _, self.greedy_done = self.greedy_approx.run()
            self.best = self.greedy_approx.get_route()

        else:
            self.opt_done = self.opt_2()

        return calc_fitness_memo(self.best), self.greedy_done and self.opt_done
    
    def opt_2(self) -> bool:
        """ Checks whether a single subsegment reversal is an improvement

        Returns:
            bool: Whether any changes were made
        """
        changed = False
        for i, j in self.subsegments:
            if self.reversal_is_improvement(i, j):
                self.best[i:j] = reversed(self.best[i:j])
                changed = True

        return not changed
    
    def reversal_is_improvement(self, i: int, j: int) -> bool:
        """ Determines if reversal of a subsegment would improve the route

        Args:
            i (int): index of first city in subsegment
            j (int): index of last city in subsegment

        Returns:
            bool: If a reversal of the subsegment results in an improvement
        """
        a, b, c, d = self.best[i-1], self.best[i], self.best[j-1], self.best[j % len(self.best)]
        return calc_distance(a, b) + calc_distance(c, d) > calc_distance(a, c) + calc_distance(b, d)
    
    def get_subsegments(self) -> tuple[tuple[int, int]]:
        """ Return (i, j) index pairs denoting tour[i:j] subsegments of a tour of length N

        Returns:
            tuple[tuple[int, int]]: subsegment of length N
        """
        return tuple((i, i + length)
                    for length in reversed(range(2, len(self.best) - 1))
                    for i in range(len(self.best) - length))
    
    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        # Draw MST edges while greedy approx is iterating
        if not self.greedy_done:
            draw_edges(window, self.greedy_approx.endpoints)

        # Draw full route when greedy approx is complete
        else:
            draw_route(window, self.best)

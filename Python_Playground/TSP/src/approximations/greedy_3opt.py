# https://colab.research.google.com/github/norvig/pytudes/blob/main/ipynb/TSP.ipynb#scrollTo=PBZUC3S6aUet
# http://matejgazda.com/tsp-algorithms-2-opt-3-opt-in-python/#:~:text=3%2Dopt%20algorithm%20works%20in,shown%20on%20the%20following%20figure.

from .approximation import Approximation
from .greedy_2opt import Opt2
from .approximation_utils import draw_route, calc_distance, calc_route_distance


class Opt3(Approximation):
    def __init__(self, cities: list) -> None:
        self.greedy_2_opt_approx = Opt2(cities)
        self.best = cities
        self.greedy_2_opt_done = False
        self.opt_done = False
        self.subsegments = self.get_subsegments()

    def get_subsegments(self) -> tuple[tuple[int, int, int]]:
        """ Return (i, j) index pairs denoting tour[i:j] subsegments of a tour of length N

        Returns:
            tuple[tuple[int, int]]: subsegment of length N
        """

        N = len(self.best)

        return ((i, j, k) for i in range(N) for j in range(i + 2, N-1) for k in range(j + 2, N - 1 + (i > 0)))

    def run(self) -> tuple[float, bool]:
        """ Perform greedy heuristic to seed initial route, then iterate through route to find any possible improvements

        Returns:
            tuple[float, bool]: returns the length of the current route and whether the approximation is completed
        """

        if not self.greedy_2_opt_done:
            _, self.greedy_2_opt_done = self.greedy_2_opt_approx.run()
            self.best = self.greedy_2_opt_approx.get_route()

        else:
            self.opt_done = self.opt_3()

        return calc_route_distance(self.best), self.greedy_2_opt_done and self.opt_done
    
    def opt_3(self) -> bool:
        """ Checks whether a single subsegment reversal is an improvement

        Returns:
            bool: Whether any changes were made
        """

        changed = False
        for i, j, k in self.subsegments:
            changed, self.best = self.get_best_case(i, j, k)
            if changed:
                break

        return not changed
    
    def get_best_case(self, i: int, j: int, k: int) -> tuple[bool, list]:
        """ Determines the best combination of 3-opt subsegments

        Args:
            i (int): index of first city in first subsegment
            j (int): index of first city in second subsegment
            k (int): index of first city in third subsegment

        Returns:
            tuple[bool, list]: If the best combination is not the original combination, the best combination
        """
        
        case_costs = {}
        for case in range(8):
            case_costs[case] = self.get_solution_cost(case, i, j, k)

        best_case = max(case_costs, key=case_costs.get)
        # If there is an improvement
        if case_costs[best_case] > 0:
            return True, self.reverse_segments(best_case, i, j, k)
        else:
            return False, self.best
    
    def get_solution_cost(self, case: int, i: int, j: int, k: int) -> float:
        a, b, c, d, e, f = self.best[i-1], self.best[i], self.best[j-1], self.best[j], self.best[k-1], self.best[k % len(self.best)]

        if case == 0:
            # Current solution
            return 0
        elif case == 1:
            # A'BC
            return calc_distance(a, b) + calc_distance(e, f) - (calc_distance(b, f) + calc_distance(a, e))
        elif case == 2:
            # ABC'
            return calc_distance(c, d) + calc_distance(e, f) - (calc_distance(d, f) + calc_distance(c, e))
        elif case == 3:
            # A'BC'
            return calc_distance(a, b) + calc_distance(c, d) + calc_distance(e, f) - (calc_distance(a, d) + calc_distance(b, f) + calc_distance(e, c))
        elif case == 4:
            # A'B'C
            return calc_distance(a, b) + calc_distance(c, d) + calc_distance(e, f) - (calc_distance(c, f) + calc_distance(b, d) + calc_distance(e, a))
        elif case == 5:
            # AB'C
            return calc_distance(b, a) + calc_distance(d, c) - (calc_distance(c, a) + calc_distance(b, d))
        elif case == 6:
            # AB'C'
            return calc_distance(a, b) + calc_distance(c, d) + calc_distance(e, f) - (calc_distance(b, e) + calc_distance(d, f) + calc_distance(c, a))
        elif case == 7:
            # A'B'C
            return calc_distance(a, b) + calc_distance(c, d) + calc_distance(e, f) - (calc_distance(a, d) + calc_distance(c, f) + calc_distance(b, e))
        
    def reverse_segments(self, case: int, i: int, j: int, k: int) -> list:
        if (i - 1) < (k % len(self.best)):
            first_segment = self.best[k % len(self.best):] + self.best[:i]
        else:
            first_segment = self.best[k % len(self.best):i]
        second_segment = self.best[i:j]
        third_segment = self.best[j:k]

        if case == 0:
            # Current solution
            return self.best
        elif case == 1:
            # A'BC
            return list(reversed(first_segment)) + second_segment + third_segment
        elif case == 2:
            # ABC'
            return first_segment + second_segment + list(reversed(third_segment))
        elif case == 3:
            # A'BC'
            return list(reversed(first_segment)) + second_segment + list(reversed(third_segment))
        elif case == 4:
            # A'B'C
            return list(reversed(first_segment)) + list(reversed(second_segment)) + third_segment
        elif case == 5:
            # AB'C
            return first_segment + list(reversed(second_segment)) + third_segment
        elif case == 6:
            # AB'C'
            return first_segment + list(reversed(second_segment)) + list(reversed(third_segment))
        elif case == 7:
            # A'B'C
            return list(reversed(first_segment)) + list(reversed(second_segment)) + list(reversed(third_segment))
    
    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        # Draw MST edges while greedy approx is iterating
        if not self.greedy_2_opt_done:
            self.greedy_2_opt_approx.draw(window)

        # Draw full route when greedy approx is complete
        else:
            draw_route(window, self.best)

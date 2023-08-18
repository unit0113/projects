import random

from .approximation import Approximation
from .approximation_utils import draw_route, calc_route_distance, randomize_route


class Star:
    def __init__(self, route: list) -> None:
        self.route = route
        self.distance = calc_route_distance(self.route)

    def update(self, black_hole: list) -> None:
        """ Moves star towards global best (black hole) via crossover
        """

        geneA = int(random.random() * len(black_hole))
        geneB = int(random.random() * len(black_hole))
        
        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        childP1 = [black_hole[i] for i in range(startGene, endGene)]    
        childP2 = [item for item in self.route if item not in childP1]

        # Randomly reverse crossover sequence
        if random.random() < 0.5:
            childP1.reverse()

        new_dist = calc_route_distance(childP1+childP2)
        if new_dist < self.distance:
            self.route = childP1+childP2
            self.distance = new_dist

    def reset(self) -> None:
        """ Explores new route if Star is swallowed by black hole
        """
        random.shuffle(self.route)


class BlackHoleOptimization(Approximation):
    def __init__(self, cities: list, num_stars: int=200, num_iterations: int=1000) -> None:
        self.num_iterations = num_iterations
        self.current_iteration = 0
        self.stars = [Star(randomize_route(cities)) for _ in range(num_stars)]
        self.sort_stars()

    def run(self) -> tuple[float, bool]:
        """Perform an iteration of particle swarm optimization

        Returns:
            tuple[float, bool]: The length of the current route and whether the approximation is completed
        """

        # Update personal bests for particles
        for star in self.stars:
            star.update(self.best.route)

        # Find global best and calculate event horizon
        self.sort_stars()
        black_hole_val = self.best.distance
        event_horizon = black_hole_val / sum([star.distance for star in self.stars[1:]])

        # Swallow stars
        for star in self.stars[1:]:
            if abs(star.distance - black_hole_val) < event_horizon:
                star.reset()

        self.current_iteration += 1
        return self.best.distance, self.current_iteration >= self.num_iterations
    
    def sort_stars(self) -> None:
        """ Sorts stars list by distance
        """
        self.stars.sort(key=lambda star: star.distance)
    
    @property
    def best(self) -> Star:
        """ Returns the currently best performing star

        Returns:
            Star: The star with the best route
        """
        return self.stars[0]

    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        draw_route(window, self.best.route)

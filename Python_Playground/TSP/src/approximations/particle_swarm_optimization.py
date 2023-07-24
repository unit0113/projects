import random

from .approximation import Approximation
from .approximation_utils import draw_route, calc_fitness_memo, randomize_route


class Particle:
    def __init__(self, route) -> None:
        self.route = route
        self.p_best = route
        self.exploration_rate = 0.02
        self.velocity = []

    def update(self) -> None:
        """ Update stored best values, and wipe old velocities. Preps particle for new iteration
        """

        self.p_best = max(self.route, self.p_best, key=calc_fitness_memo)
        self.velocity.clear()

        '''for swapped in range(len(self.route)):
            if random.random() < self.exploration_rate:
                swapWith = int(random.random() * len(self.route))
                self.velocity.append((swapped, swapWith, 1))'''
        
        for _ in range(int(self.exploration_rate * random.random() * len(self.route))):
            swap, swap_with = random.sample([*range(len(self.route))], 2)
            self.velocity.append((swap, swap_with, 1))

        self.exploration_rate *= 0.999

    def add_velocity(self, swap: tuple[int, int, float]) -> None:
        """ Add potential swap to velocity

        Args:
            swap (tuple[int, int, float]): index position of city in current route, index position of city in best route, probability of swap occuring
        """

        self.velocity.append(swap)

    def update_position(self) -> None:
        for swap in self.velocity:
            if random.random() < swap[2]:
                self.route[swap[0]], self.route[swap[1]] = self.route[swap[1]], self.route[swap[0]]


class ParticleSwarmOptimization(Approximation):
    def __init__(self, cities: list, swarm_size: int=200, num_iterations: int=500, global_best_prob: float= 0.10, personal_best_prob: float= 0.05) -> None:
        self.num_cities = len(cities)
        self.swarm_size = swarm_size
        self.num_iterations = num_iterations
        self.current_iteration = 0
        self.particles = [Particle(randomize_route(cities)) for _ in range(swarm_size)]
        self.global_best_prob = global_best_prob
        self.personal_best_prob = personal_best_prob

        self.best = self._set_global_max()

    def _set_global_max(self) -> None:
        self.best = max(self.particles, key=lambda p: calc_fitness_memo(p.route)).route

    def run(self) -> tuple[float, bool]:
        """Perform an iteration of particle swarm optimization

        Returns:
            tuple[float, bool]: The score of the current route and whether the approximation is completed
        """

        # Update personal bests for particles
        for particle in self.particles:
            particle.update()

        # Calculate velocities and update particle positions
        for particle in self.particles:
            if particle.route == self.best:
                continue

            # Update velocities with respect to personal best
            for index in range(len(particle.route)):
                if particle.route[index] != particle.p_best[index]:
                    particle.add_velocity((index, particle.p_best.index(particle.route[index]), self.personal_best_prob))

            # Update velocities with respect to global best
            for index in range(len(particle.route)):
                if particle.route[index] != self.best[index]:
                    particle.add_velocity((index, self.best.index(particle.route[index]), self.global_best_prob))

            # Update position based on velocites
            particle.update_position()

        # Find global best and update probabilities
        self._set_global_max()
        self.personal_best_prob *= 0.99
        self.global_best_prob *= 1.02

        self.current_iteration += 1
        return calc_fitness_memo(self.best), self.current_iteration >= self.num_iterations

    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        draw_route(window, self.best)
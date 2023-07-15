from genetic_tsp_class import GeneticAlgorithm
from utils import City, calc_fitness_memo

class ParallelGA:
    def __init__(self, init_population: list[City], num_populations: int=5, num_generations: int=500) -> None:
        self.populations = [GeneticAlgorithm(init_population) for _ in range(num_populations)]
        self.current_generation = 0
        self.num_generations = num_generations

    @property
    def best(self):
        return max([population.best for population in self.populations], key = calc_fitness_memo)
    
    def evolve(self) -> tuple[float, bool]:
        """Perform a single step in the genetic process for each population

        Returns:
            tuple[float, bool]: returns the score of the top performing organism and whether the approximation is completed
        """

        for population in self.populations:
            population._evolve_next_generation()
        
        self.current_generation += 1
        return calc_fitness_memo(self.best), self.current_generation >= self.num_generations
    
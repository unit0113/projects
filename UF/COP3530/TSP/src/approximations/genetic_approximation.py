import numpy as np
import random
import pandas as pd
import sys

from .approximation import Approximation
sys.path.append('..')
from TSP.src import functions


class GeneticApproximation(Approximation):
    def __init__(self, init_population, pop_size, elite_size, mutation_rate, num_generations) -> None:
        self.population = [functions.randomize_route(init_population) for _ in range(pop_size)]
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.current_generation = 0
        self.num_generations = num_generations
        self.best = None

    def run(self) -> tuple[list, bool]:
        """Perform a single step in the genetic process

        Returns:
            tuple[list, bool]: returns the top performing organism and whether the approximation is completed
        """

        self._evolve_next_generation()
        return self.best, self.current_generation >= self.num_generations

    def _evolve_next_generation(self) -> None:
        """Runs algorithm through the genetic evolution process
           First: Ranks and sorts the current population
           Performs fitness proportionate selection on the population
           Breeds and mutatates population
        """

        pop_ranked = self._rank_pops()
        self.best = pop_ranked[0][0]
        selection_results = self._selection_FPS(pop_ranked)
        children = self._breed_population(selection_results)
        self.population = self._mutate_population(children)
        self.current_generation += 1

    def _rank_pops(self) -> list[tuple[list, float]]:
        """Creates a sorted list of tuples containing the population and its fitness

        Returns:
            list: list of two element tuples, containing the population and its fitness
        """

        return sorted([(pop, functions.calc_fitness_memo(pop)) for pop in self.population], key=lambda x: x[1], reverse = True)

    def _selection_FPS(self, ranked_pops: list[tuple[list, float]]) -> list:
        """Selects populations for the mating pool via fitness proportionate selection. Top populations proceed based on elite size value.
           Remaining populations are selected via weighted random sampling.

        Args:
            ranked_pops (list[tuple[list, float]]): list of two element tuples, containing the population and its fitness

        Returns:
            list: selected populations for the mating pool
        """

        # Specified number of top-ranked populations continue to next generation
        selection_results = [pop for pop, score in ranked_pops[:self.elite_size]]
        ranked_pops_weights = [weight for pop, weight in ranked_pops]

        df = pd.DataFrame(np.array(ranked_pops_weights), columns=["Fitness"])
        df['cum_perc'] = 100 * df.Fitness.cumsum() / df.Fitness.sum()

        for _ in range(0, len(ranked_pops) - self.elite_size):
            for i in range(0, len(ranked_pops)):
                if 100 * random.random() <= df.iat[i,1]:
                    selection_results.append(ranked_pops[i][0])
                    break

        return selection_results

    def _breed_population(self, mating_pool: list[list]) -> list:
        """Breeds the mating pool to create children using ordered crossover

        Args:
            mating_pool (list): selected populations for the mating pool

        Returns:
            list: children that result from the breeding process
        """

        children = [pop for pop in mating_pool[:self.elite_size]]

        pool = random.sample(mating_pool, len(mating_pool))        
        for index, pop in enumerate(pool[:-self.elite_size]):
            children.append(self._breed(pop, pool[len(mating_pool) - index - 1]))

        return children

    def _breed(self, parent1: list, parent2: list) -> list:
        """Breeds two parents and creates child

        Args:
            parent1 (list): 1st parent
            parent2 (list): 2nd parent

        Returns:
            list: result of breeding process
        """

        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))
        
        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        childP1 = [parent1[i] for i in range(startGene, endGene)]    
        childP2 = [item for item in parent2 if item not in childP1]

        return childP1 + childP2

    def _mutate_population(self, population: list[list]) -> list:
        """Mutates population

        Args:
            population (list): Current population

        Returns:
            list: mutated population
        """

        mutated_population = [pop for pop in population[:self.elite_size]]
        mutated_population.extend([self._mutate(population[i]) for i in range(self.elite_size,len(population))])
        return mutated_population

    def _mutate(self, individual: list) -> list:
        """Mutates a selected individual

        Args:
            individual (list): individual to be mutated

        Returns:
            list: a mutated individual
        """
        
        for swapped in range(len(individual)):
            if random.random() < self.mutation_rate:
                swapWith = int(random.random() * len(individual))
                individual[swapped], individual[swapWith] = individual[swapWith], individual[swapped]

        return individual

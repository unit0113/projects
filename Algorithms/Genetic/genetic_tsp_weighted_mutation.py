import numpy as np
import random
import pandas as pd

from utils import City, calc_fitness_memo, create_individual, calc_distance


class GeneticAlgorithmWeightedMutation:
    def __init__(self, init_population: list[City], pop_size: int=100, elite_size: int=10, mutation_rate: float=0.00001, num_generations: int=250) -> None:
        self.pop_size = pop_size
        self.init_population = init_population
        self.population = [create_individual(init_population) for _ in range(self.pop_size)]
        self.ranked_pops = self._rank_pops(self.population)
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.current_generation = 0
        self.num_generations = num_generations
        self.best = self.population[0]

    def _sort_population(self, population):
        return sorted(population, key=calc_fitness_memo, reverse=True)

    def evolve(self) -> tuple[float, bool]:
        """Perform a single step in the genetic process

        Returns:
            tuple[float, bool]: returns the score of the top performing organism and whether the approximation is completed
        """

        self._evolve_next_generation()
        return calc_fitness_memo(self.best), self.current_generation >= self.num_generations

    def _evolve_next_generation(self) -> None:
        """Runs algorithm through the genetic evolution process
           First: Ranks and sorts the current population
           Performs fitness proportionate selection on the population
           Breeds and mutatates population
        """

        elite = self.population[:self.elite_size]
        selection_results = self._selection_FPS()
        children = self._breed_population(selection_results)
        self.ranked_pops = self._rank_pops(self._mutate_population(children) + elite)[:self.pop_size]
        self.population = [pop for pop, weight in self.ranked_pops]
        self.best = self.population[0]
        self.current_generation += 1

    def _rank_pops(self, population) -> list[tuple[list, float]]:
        """Creates a sorted list of tuples containing the population and its fitness

        Returns:
            list: list of two element tuples, containing the population and its fitness
        """
        return sorted([(pop, calc_fitness_memo(pop)) for pop in population], key=lambda x: x[1], reverse = True)
    
    def _selection_FPS(self) -> list:
        """Selects populations for the mating pool via fitness proportionate selection. Top populations proceed based on elite size value.
           Remaining populations are selected via weighted random sampling.

        Args:
            ranked_pops (list[tuple[list, float]]): list of two element tuples, containing the population and its fitness

        Returns:
            list: selected populations for the mating pool
        """

        # Specified number of top-ranked populations continue to next generation
        selection_results = []
        ranked_pops_weights = [weight for pop, weight in self.ranked_pops]

        df = pd.DataFrame(np.array(ranked_pops_weights), columns=["Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

        for _ in range(0, self.pop_size):
            for i in range(0, self.pop_size):
                if 100 * random.random() <= df.iat[i,2]:
                    selection_results.append(self.ranked_pops[i][0])
                    break

        return selection_results

    def _breed_population(self, mating_pool: list) -> list:
        """Breeds the mating pool to create children using ordered crossover
        
        Args:
            mating_pool (list): selected populations for the mating pool

        Returns:
            list: children that result from the breeding process
        """

        pool = random.sample(mating_pool, self.pop_size)
        return [self._breed(pop, pool[self.pop_size - index - 1]) for index, pop in enumerate(pool)]        

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
            population (list): Population to be mutated

        Returns:
            list: mutated population
        """

        return [self._mutate(population[i]) for i in range(len(population))]

    def _mutate(self, individual: list) -> list:
        """Mutates a selected individual

        Args:
            individual (list): individual to be mutated

        Returns:
            list: a mutated individual
        """
        
        for swapped in range(len(individual)):
            dist_score = calc_distance(individual[swapped], individual[swapped+1]) if swapped < len(individual) - 1 else calc_distance(individual[swapped], individual[0])
            if random.random() < self.mutation_rate * dist_score:
                swapWith = int(random.random() * len(individual))
                individual[swapped], individual[swapWith] = individual[swapWith], individual[swapped]

        return individual

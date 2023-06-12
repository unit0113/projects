import numpy as np
import random
import operator
import pandas as pd

from src import functions


class GeneticApproximation:
    def __init__(self, init_population, pop_size, elite_size, mutation_rate, num_generations, plot = True) -> None:
        self.population = [functions.createRandomRoute(init_population) for _ in range(pop_size)]
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.progress = [1 / self.rank_pops()[0][1]]
        self.plot = plot

    def evolve(self):
        for _ in range(self.num_generations):
            self.evolve_next_generation()
            self.progress.append(1 / self.rank_pops()[0][1])

    def evolve_next_generation(self):
        pop_ranked = self.rank_pops()
        selection_results = self.selection(pop_ranked)
        mating_pool = self.create_mating_pool(selection_results)
        children = self.breed_population(mating_pool)
        self.population = self.mutatePopulation(children)

    def rank_pops(self):
        fitnessResults = {}
        for i in range(len(self.population)):
            fitnessResults[i] = functions.calc_fitness(self.population[i])
        return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

    def selection(self, pop_ranked):
        selectionResults = []
        df = pd.DataFrame(np.array(pop_ranked), columns=["Index","Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

        selectionResults = [pop_ranked[i][0] for i in range(self.elite_size)]

        for _ in range(0, len(pop_ranked) - self.elite_size):
            for i in range(0, len(pop_ranked)):
                if 100 * random.random() <= df.iat[i,3]:
                    selectionResults.append(pop_ranked[i][0])
                    break

        return selectionResults

    def create_mating_pool(self, selectionResults):
        return [self.population[index] for index in selectionResults]

    def breed_population(self, mating_pool):
        length = len(mating_pool) - self.elite_size
        pool = random.sample(mating_pool, len(mating_pool))

        children = [mating_pool[i] for i in range(self.elite_size)]
        
        for i in range(length):
            children.append(self._breed(pool[i], pool[len(mating_pool) - i - 1]))

        return children

    def _breed(self, parent1, parent2):
        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))
        
        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        childP1 = [parent1[i] for i in range(startGene, endGene)]    
        childP2 = [item for item in parent2 if item not in childP1]

        return childP1 + childP2

    def mutatePopulation(self, population):
        return [self._mutate(population[i]) for i in range(len(population))]

    def _mutate(self, individual):
        for swapped in range(len(individual)):
            if random.random() < self.mutation_rate:
                swapWith = int(random.random() * len(individual))
                individual[swapped], individual[swapWith] = individual[swapWith], individual[swapped]

        return individual

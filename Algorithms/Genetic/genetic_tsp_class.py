import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import time


class City:
    def __init__(self, map_size):
        self.x = int(random.random() * map_size)
        self.y = int(random.random() * map_size)
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class GeneticAlgorithm:
    def __init__(self, init_population, pop_size, elite_size, mutation_rate, num_generations, plot = True) -> None:
        self.pop_size = pop_size
        self.population = [self.createRoute(init_population) for _ in range(self.pop_size)]
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.progress = [1 / self.rank_pops()[0][1]]
        self.plot = plot

    def createRoute(self, gene_list):
        return random.sample(gene_list, len(gene_list))

    def evolve(self):
        start = time.time()
        for _ in range(self.num_generations):
            self.evolve_next_generation2()

        print(time.time() - start)

        if self.plot:
            plt.plot(self.progress)
            plt.ylabel('Distance')
            plt.xlabel('Generation')
            plt.show()

    def evolve_next_generation(self):
        pop_ranked = self.rank_pops()
        self.progress.append(1 / pop_ranked[0][1])
        selection_results = self.selection(pop_ranked)
        mating_pool = self.create_mating_pool(selection_results)
        children = self.breed_population(mating_pool)
        self.population = self.mutatePopulation(children)

    def rank_pops(self):
        fitnessResults = {}
        for index, pop in enumerate(self.population):
            fitnessResults[index] = self.calc_fitness(pop)
        return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

    def calc_fitness(self, organism):
        distance = organism[0].distance(organism[-1])
        for i, start_organism in enumerate(organism[:-1]):
            end_organism = organism[i + 1]                
            distance += start_organism.distance(end_organism)

        return 1 / distance

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
    
    def evolve_next_generation2(self):
        pop_ranked = self.rank_pops2()
        self.progress.append(1 / pop_ranked[0][1])
        mating_pool = self.selection_FPS(pop_ranked)
        children = self.breed_population2(mating_pool)
        self.population = self.mutatePopulation(children)
    
    def rank_pops2(self) -> list[tuple[list, float]]:
        """Creates a sorted list of tuples containing the population and its fitness

        Returns:
            list: list of two element tuples, containing the population and its fitness
        """
        return sorted([(pop, self.calc_fitness(pop)) for pop in self.population], key=lambda x: x[1], reverse = True)
    
    def selection_FPS(self, ranked_pops: list[tuple[list, float]]) -> list:
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
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

        for _ in range(0, len(ranked_pops) - self.elite_size):
            for i in range(0, len(ranked_pops)):
                if 100 * random.random() <= df.iat[i,2]:
                    selection_results.append(ranked_pops[i][0])
                    break

        return selection_results

    def breed_population2(self, mating_pool: list) -> list:
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


if __name__ == "__main__":
    num_cities = 25
    map_size = 200
    city_list = [City(map_size) for _ in range(num_cities)]

    genetic_tsp = GeneticAlgorithm(city_list, 100, 20, 0.01, 500)
    genetic_tsp.evolve()

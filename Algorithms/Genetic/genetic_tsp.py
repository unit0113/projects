#https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35


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


def calc_fitness(route):
    distance = 0
    for i, fromCity in enumerate(route):
        if i + 1 < len(route):
            toCity = route[i + 1]
        else:
            toCity = route[0]
        distance += fromCity.distance(toCity)

    return 1 / distance


def createRoute(cityList):
    return random.sample(cityList, len(cityList))


def initialPopulation(popSize, cityList):
    return [createRoute(cityList) for _ in range(popSize)]


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = calc_fitness(population[i])
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])

    for _ in range(0, len(popRanked) - eliteSize):
        for i in range(0, len(popRanked)):
            if 100 * random.random() <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break

    return selectionResults


def create_mating_pool(population, selectionResults):
    return [population[index] for index in selectionResults]


def breed(parent1, parent2):
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    childP1 = [parent1[i] for i in range(startGene, endGene)]    
    childP2 = [item for item in parent2 if item not in childP1]

    return childP1 + childP2


def breed_population(matingpool, elite_size):
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))

    children = [matingpool[i] for i in range(elite_size)]
    
    for i in range(length):
        children.append(breed(pool[i], pool[len(matingpool) - i - 1]))

    return children


def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swapWith = int(random.random() * len(individual))
            individual[swapped], individual[swapWith] = individual[swapWith], individual[swapped]

    return individual


def mutatePopulation(population, mutation_rate):
    return [mutate(population[i], mutation_rate) for i in range(len(population))]


def evolve_next_generation(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = create_mating_pool(currentGen, selectionResults)
    children = breed_population(matingpool, eliteSize)
    return mutatePopulation(children, mutationRate)


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for _ in range(0, generations):
        pop = evolve_next_generation(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    start = time.time()

    for _ in range(0, generations):
        pop = evolve_next_generation(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    print(time.time() - start)
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


if __name__ == "__main__":
    num_cities = 25
    map_size = 200
    city_list = [City(map_size) for _ in range(num_cities)]

    geneticAlgorithmPlot(population=city_list, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
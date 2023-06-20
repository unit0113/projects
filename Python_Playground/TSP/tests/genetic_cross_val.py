import itertools
import random
import numpy as np
import time
import sys

# Allows importing from directories higher in file path
sys.path.append('..')
from TSP.src.approximations.genetic_approximation import GeneticApproximation
from TSP.src import functions


# City class with random initilization
class City:
    def __init__(self, map_size):
        self.x = int(random.random() * map_size)
        self.y = int(random.random() * map_size)
    
    def distance_from(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class GeneticCrossVal:
    def __init__(self, population: list, param_grid: dict) -> dict:
        self.population = population
        self.param_grid = param_grid

    def fit(self) -> list[tuple[dict, float, float]]:
        """Performs grid search cross validation on the provided paramater grid

        Returns:
            list: List containing tuples of the paramater set, fitness score, and duration of the test cycle for that paramater set
        """

        param_scores = []

        # Perform grid search cross val over all combinations of params
        for params in itertools.product(*self.param_grid.values()):
            run_scores = []

            # Repeat 5 times to get average and account for randomness
            start = time.time()
            for _ in range(5):
                gen_approx = GeneticApproximation(self.population, *params)
                done = False
                while not done:
                    best, done = gen_approx.run()
                    run_scores.append(functions.calc_fitness(best))
            
            duration = time.time() - start

            param_scores.append((params, sum(run_scores), duration))

        return sorted(param_scores, key = lambda x: x[1], reverse = True)
            

if __name__ == "__main__":
    num_cities = 25
    map_size = 200
    city_list = [City(map_size) for _ in range(num_cities)]

    params = {'pop_size': [50, 100, 150, 200, 250], 'elite_size': [10, 20, 30, 40, 50], 'mutation_rate': [0.001, 0.01, 0.1, 0.2, 0.5], 'num_generations': [250]}
    genetic_cv = GeneticCrossVal(city_list, params)
    param_scores = genetic_cv.fit()
    print(param_scores[:20])


# Results: 19 Jun, 25 cities
"""[((250, 10, 0.001, 250), 1.2758261618221556, 69.74723291397095),
   ((200, 40, 0.001, 250), 1.272967583052799, 45.812567949295044),
   ((250, 20, 0.001, 250), 1.267630644322337, 68.94131994247437),
   ((250, 30, 0.001, 250), 1.2629407809680888, 66.27685284614563),
   ((200, 20, 0.001, 250), 1.2534909655376318, 49.896891832351685),
   ((250, 40, 0.001, 250), 1.2520705225555897, 64.45660543441772),
   ((200, 10, 0.001, 250), 1.2488410738703255, 52.04733395576477),
   ((200, 30, 0.001, 250), 1.2460389083180676, 47.402432441711426),
   ((200, 50, 0.001, 250), 1.2459801423370476, 43.983081340789795),
   ((150, 40, 0.001, 250), 1.2320677729675493, 30.32062578201294),
   ((250, 10, 0.01, 250), 1.228139715864845, 71.1129162311554),
   ((100, 10, 0.001, 250), 1.2275070213526698, 19.540728330612183),
   ((100, 30, 0.001, 250), 1.2234200830809803, 16.682739973068237),
   ((150, 10, 0.001, 250), 1.2225501319403524, 34.2347252368927),
   ((250, 20, 0.01, 250), 1.2179754585612779, 69.14761781692505),
   ((250, 30, 0.01, 250), 1.217845157411731, 66.25005531311035),
   ((250, 40, 0.01, 250), 1.2143402028450498, 63.4615523815155),
   ((150, 50, 0.001, 250), 1.2122591036942596, 28.029962301254272),
   ((250, 50, 0.001, 250), 1.2087984121010844, 61.70221996307373),
   ((200, 40, 0.01, 250), 1.2070118375553733, 44.761268615722656)]"""
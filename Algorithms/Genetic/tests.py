import matplotlib.pyplot as plt
import time

from utils import City
from genetic_tsp_class import GeneticAlgorithm
from genetic_tsp_migration import GeneticAlgorithmMigration
from genetic_tsp_weighted_mutation import GeneticAlgorithmWeightedMutation


if __name__ == '__main__':
    num_cities = 200
    map_size = 200
    city_list = [City(map_size) for _ in range(num_cities)]

    approximations = [GeneticAlgorithm, GeneticAlgorithmMigration, GeneticAlgorithmWeightedMutation]
    names = []
    for approx in approximations:
        approx = approx(city_list, num_generations=500)
        names.append(approx.__class__.__name__)
        start = time.time()
        done = False
        scores = []
        while not done:
            best, done = approx.evolve()
            scores.append(1 / best)

        print(f'{names[-1]}: {str(time.time() - start)}')
        plt.plot(scores)

    plt.ylabel('Distance')
    plt.xlabel('Generation') 
    plt.legend(names)
    plt.show()
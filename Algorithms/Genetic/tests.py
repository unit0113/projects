import matplotlib.pyplot as plt
import time

from utils import City
from genetic_tsp_class import GeneticAlgorithm
from genetic_tsp_migration import GeneticAlgorithmMigration
from genetic_tsp_weighted_mutation import GeneticAlgorithmWeightedMutation
from parallel_genetic import ParallelGA
from genetic_new_mutations import GeneticAlgorithmNewMutations
from genetic_tsp_tournement import GeneticAlgorithmTournement


if __name__ == '__main__':
    num_cities = 200
    map_size = 200
    city_list = [City(map_size) for _ in range(num_cities)]

    approximations = [GeneticAlgorithm, GeneticAlgorithmTournement]
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





"""TODO
   -Migration?
   -Adaptive mutation rate?
   -Common gene detection?
   -Tournament selection

   -Selection
   -Crossover
   -Mutation

"""
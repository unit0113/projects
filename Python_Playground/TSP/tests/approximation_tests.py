import sys
import matplotlib.pyplot as plt

from random_city import City

# Allows importing from directories higher in file path
sys.path.append('..')
from TSP.src.approximations.genetic_approximation import GeneticApproximation
from TSP.src.functions import calc_fitness


def genetic_test(city_list):
    genetic_approx = GeneticApproximation(city_list, 200, 10, 0.001, 500)
    done = False

    scores = []
    while not done:
        best, done = genetic_approx.run()
        scores.append(1 / calc_fitness(best))

    plt.plot(scores)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


if __name__ == '__main__':
    num_cities = 100
    map_size = 200
    city_list = [City(map_size) for _ in range(num_cities)]

    genetic_test(city_list)

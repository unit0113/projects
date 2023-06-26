import sys
import matplotlib.pyplot as plt
import time

from random_city import City

# Allows importing from directories higher in file path
sys.path.append('..')
from TSP.src.approximations.brute_force import BruteForce
from TSP.src.approximations.genetic_approximation import GeneticApproximation
from TSP.src.functions import calc_fitness, calc_fitness_memo


def memoization_test(city_list):
    start = time.time()
    for _ in range(1000):
        dist = calc_fitness(city_list)
    print(dist, time.time() - start)

    start = time.time()
    for _ in range(1000):
        dist = calc_fitness_memo(city_list)
    print(dist, time.time() - start)


def plot(brute_force_score, scores):
    plt.plot(scores)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.axhline(y = brute_force_score, color = 'r', linestyle = 'dashed')  
    plt.show()


def brute_force(city_list):
    brute_force_approx = BruteForce(city_list)
    done = False
    while not done:
        best, done = brute_force_approx.run()

    return 1 / calc_fitness_memo(best)


def genetic_test(city_list, brute_force_score):
    start = time.time()

    genetic_approx = GeneticApproximation(city_list, 200, 10, 0.001, 500)

    done = False
    scores = []
    while not done:
        best, done = genetic_approx.run()
        scores.append(1 / calc_fitness_memo(best))

    print('Genetic run time: ' + str(time.time() - start))
    plot(brute_force_score, scores)


if __name__ == '__main__':
    num_cities = 200
    map_size = 200
    city_list = [City(map_size) for _ in range(num_cities)]

    brute_force_score = 0
    #brute_force_score = brute_force(city_list)
    genetic_test(city_list, brute_force_score)

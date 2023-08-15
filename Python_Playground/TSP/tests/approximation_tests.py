import matplotlib.pyplot as plt
import time

from random_city import RandomCity

# Allow import from src folder
from sys import path
from os import getcwd
path.insert(0, getcwd())
from src.approximations import NearestNeighbor, Greedy, Opt2, Opt3, DivideAndConquer, Cristofides, Genetic, SimmulatedAnnealing, AntColonyOptimization, BeeColonyOptimization, ParticleSwarmOptimization, BruteForce


if __name__ == '__main__':
    # Counteracts the matplotlib.use('Agg') in run_state which allows drawing to pygame surface
    import matplotlib
    matplotlib.use('tkagg')

    num_cities = 200
    map_size = 200
    city_list = [RandomCity(map_size) for _ in range(num_cities)]

    approximations = [BeeColonyOptimization]
    names = []
    plt.figure(figsize=(20, 12), dpi=80)
    for approx in approximations:
        approx = approx(city_list)
        names.append(approx.__class__.__name__)
        start = time.time()
        done = False
        scores = []
        while not done:
            best, done = approx.run()
            scores.append(best)

        print(f'{names[-1]}: {str(time.time() - start)}')
        plt.plot(scores)

    plt.ylabel('Distance')
    plt.xlabel('Iteration') 
    plt.xscale('log')
    plt.legend(names)
    plt.show()

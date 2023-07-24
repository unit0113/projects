import matplotlib.pyplot as plt
import time

from random_city import RandomCity

# Allow import from src folder
from sys import path
from os.path import dirname
path.append(dirname(path[0]))
from src.approximations.ant_colony_opimization import AntColonyOptimization


if __name__ == '__main__':
    # Counteracts the matplotlib.use('Agg') in run_state which allows drawing to pygame surface
    import matplotlib
    matplotlib.use('tkagg')

    num_cities = 200
    map_size = 200
    city_list = [RandomCity(map_size) for _ in range(num_cities)]

    vals = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 1]
    names = []
    plt.figure(figsize=(20, 12), dpi=80)
    for val in vals:
        approx = AntColonyOptimization(city_list, initial_pheromone_strength=val)
        names.append(val)
        start = time.time()
        done = False
        scores = []
        while not done:
            best, done = approx.run()
            scores.append(1 / best)

        print(f'{names[-1]}: {str(time.time() - start)}')
        plt.plot(scores)

    plt.ylabel('Distance')
    plt.xlabel('Iteration') 
    plt.xscale('log')
    plt.legend(names)
    plt.show()

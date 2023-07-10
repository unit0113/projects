import random
import numpy as np


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


def create_individual(gene_list):
        return random.sample(gene_list, len(gene_list))


#https://codereview.stackexchange.com/questions/110221/tsp-brute-force-optimization-in-python
def memoize(func: callable) -> float:
    """Create a memoization dictionary for calculating the distance between cities

    Args:
        func (callable): Distance function

    Returns:
        float: Distance between cities
    """

    class memo_dict(dict):
        def __init__(self, func):
            self.func = func
        
        def __call__(self, *args):
            return self[args]
        
        def __missing__(self, key):
            result = self[key] = self.func(*key)
            return result
    
    return memo_dict(func)


@memoize
def calc_distance(city1: City, city2: City) -> float:
    """Allows memoization of distance calculation

    Args:
        city1 (City): Start City
        city2 (City): End City

    Returns:
        float: distance between cities
    """
    dist = city1.distance_from(city2)
    return dist


def calc_fitness_memo(route: list[City]) -> float:
    """Determine the fitness of a route. Metric is 1 / total_distance, goal is to maximize the fitness. Utilizes memoization dict to reduce repeated calculations of distances

    Args:
        route (list): route of cities to be scored

    Returns:
        float: score of the input route
    """

    distance = calc_distance(route[0], route[-1])
    for i, start in enumerate(route[:-1]):
        end = route[i + 1]            
        distance += calc_distance(start, end)

    return 1 / distance

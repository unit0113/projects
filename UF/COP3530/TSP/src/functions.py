import random
import sys

# Allows importing from directories higher in file path
sys.path.append('..')
from TSP.src.city import City


def randomize_route(route: list) -> list:
    """Randomizes the input

    Args:
        route (list): list of city objects to be randomized

    Returns:
        list: suffled list of city objects
    """
    return random.sample(route, len(route))


def calc_fitness(route: list[City]) -> float:
    """Determine the fitness of a route. Metric is 1 / total_distance, goal is to maximize the fitness.

    Args:
        route (list): route of cities to be scored

    Returns:
        float: score of the input route
    """

    distance = route[0].distance_from(route[-1])
    for i, start in enumerate(route[:-1]):
        end = route[i + 1]            
        distance += start.distance_from(end)

    return 1 / distance


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


def get_cities(window, num_cities: int) -> list[City]:
    """Loads specified number of cities from file and shuffles

    Args:
        window (pygame.surface.Surface): pygame window object for City init
        num_cities (int): Number of cities to load

    Returns:
        list[City]: randomized list of cities
    """

    with open(r'assets/cities.csv', 'r') as file:
        cities = [City(window, *params.strip().split(',')) for params in file.readlines()[1:num_cities + 1]]

    random.shuffle(cities)
    return cities

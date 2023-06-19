import random

def randomize_route(route: list) -> list:
    """Randomizes the input

    Args:
        route (list): list of city objects to be randomized

    Returns:
        list: suffled list of city objects
    """
    return random.sample(route, len(route))


def calc_fitness(route: list) -> float:
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
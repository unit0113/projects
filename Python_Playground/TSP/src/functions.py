import random

def createRandomRoute(items):
    return random.sample(items, len(items))


def calc_fitness(route):
    distance = route[0].distance_from(route[-1])
    for i, start in enumerate(route[:-1]):
        end = route[i + 1]            
        distance += start.distance_from(end)

    return 1 / distance
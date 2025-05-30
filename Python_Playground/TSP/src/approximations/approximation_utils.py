import pygame
import random

from src.colors import MAXIMUM_GREEN


def randomize_route(route: list) -> list:
    """Randomizes the input

    Args:
        route (list): list of city objects to be randomized

    Returns:
        list: suffled list of city objects
    """
    return random.sample(route, len(route))


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
def calc_distance(city1, city2) -> float:
    """Allows memoization of distance calculation

    Args:
        city1 (City): Start City
        city2 (City): End City

    Returns:
        float: distance between cities
    """
    dist = city1.distance_from(city2)
    return dist


def calc_fitness_memo(route: list) -> float:
    """Determine the fitness of a route. Metric is 1 / total_distance, goal is to maximize the fitness. Utilizes memoization dict to reduce repeated calculations of distances

    Args:
        route (list): route of cities to be scored

    Returns:
        float: score of the input route
    """

    return 1 / calc_route_distance(route)


def calc_route_distance(route: list) -> float:
    """ Calculates the total distance of a particular route

    Args:
        route (list): the route

    Returns:
        float: total euclidean distance of the route
    """

    return sum(calc_distance(route[i], route[i-1]) for i in range(len(route)))


def draw_route(window: pygame.surface.Surface, route: list) -> None:
    """ Draw route when presented as sequential list of city objects

    Args:
        window (pygame.surface.Surface): Game window that route will be drawn onto
        route (list): Sequential list of city objects that represent the calculated route
    """

    if len(route) < 2:
        return

    # Connect first and last point
    pygame.draw.line(window, MAXIMUM_GREEN, route[0].get_pixel_tuple(), route[-1].get_pixel_tuple(), 2)

    # Loop through remaining connections
    for index, city in enumerate(route[1:]):
        pygame.draw.line(window, MAXIMUM_GREEN, route[index].get_pixel_tuple(), city.get_pixel_tuple(), 2)

def draw_grid(game, grid: dict[dict[float]]) -> None:
    """ Draw all connections based on relative strength of connection

    Args:
        window (pygame.surface.Surface): Game window that route will be drawn onto
        grid (dict[dict[float]]): Sequential list of city objects that represent the calculated route
    """

    max_val = max([max(innder_dict.values()) for innder_dict in grid.values()])
    x, y, height, width = game.assets['map'].get_x_y_height_width()
    route_surface = pygame.Surface((width, height)).convert_alpha()
    route_surface.fill((0, 0, 0, 0))

    for outer_city in grid.keys():
        for inner_city in grid[outer_city].keys():
            if outer_city is inner_city:
                continue
            alpha = int(255 * grid[outer_city][inner_city] / max_val)
            if alpha > 0:
                pygame.draw.line(route_surface, (*MAXIMUM_GREEN, alpha), outer_city.get_pixel_tuple(), inner_city.get_pixel_tuple(), 2)

    game.window.blit(route_surface, (x, y))

def draw_edges(window: pygame.surface.Surface, edges: dict) -> None:
    """ Draw connections based on a dict containing edges/segments

    Args:
        window (pygame.surface.Surface): Game window that route will be drawn onto
        edges (dict): Dictionary of edges to draw
    """

    seen = set()

    for endpoint in edges:
        if len(edges[endpoint]) > 1 and endpoint not in seen:
            seen.add(edges[endpoint][0])
            seen.add(edges[endpoint][-1])
            # Loop through remaining connections
            for index, city in enumerate(edges[endpoint][1:]):
                pygame.draw.line(window, MAXIMUM_GREEN, edges[endpoint][index].get_pixel_tuple(), city.get_pixel_tuple(), 2)


def draw_edge_list(window: pygame.surface.Surface, edges: list[tuple]) -> None:
    for city_a, city_b in edges:
        pygame.draw.line(window, MAXIMUM_GREEN, city_a.get_pixel_tuple(), city_b.get_pixel_tuple(), 2)


def draw_mst(window: pygame.surface.Surface, edges: dict) -> None:

    links = [(parent, child) for parent in edges for child in edges[parent]]
    for parent, child in links:
        pygame.draw.line(window, MAXIMUM_GREEN, parent.get_pixel_tuple(), child.get_pixel_tuple(), 2)
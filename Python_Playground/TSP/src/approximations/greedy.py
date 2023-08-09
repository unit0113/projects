# https://colab.research.google.com/github/norvig/pytudes/blob/main/ipynb/TSP.ipynb#scrollTo=W_tj4PHgTmZC

from itertools import combinations

from .approximation import Approximation
from .approximation_utils import draw_edges, draw_route, calc_distance, calc_fitness_memo


class Greedy(Approximation):
    def __init__(self, cities: list) -> None:
        self.num_cities = len(cities)
        self.endpoints = {city: [city] for city in cities}
        self.shortest_links = sorted(combinations(cities, 2), key=lambda link: calc_distance(*link))
        self.current_link = 0
        self.num_links = len(self.shortest_links)
        self.done = False
        self.route = self.get_route()

    def run(self) -> tuple[float, bool]:
        """Add a single edge to the current route

        Returns:
            tuple[float, bool]: returns the score of the current route and whether the approximation is completed
        """

        # Add link to solution
        joined_segment = []
        while not joined_segment:
            # Examine next best link
            city_a, city_b = self.shortest_links[self.current_link]
            if city_a in self.endpoints and city_b in self.endpoints and self.endpoints[city_a] != self.endpoints[city_b]:
                joined_segment = self.join_segments(city_a, city_b)
                if len(joined_segment) == self.num_cities:
                    self.done = True
            
            # Increment link
            self.current_link += 1

        self.route = self.get_route()
        return calc_fitness_memo(self.route), self.done
    
    def join_segments(self, city_a: 'City', city_b: 'City') -> list:
        """ Join segments [...,A] + [B,...] into one segment. Maintain `endpoints`.

        Args:
            city_a (City): City to be connected
            city_b (City): Second city to be connected

        Returns:
            list: The combined segment that results from joining the two cities
        """

        a_seg, b_seg = self.endpoints[city_a], self.endpoints[city_b]

        # Reverse as required to join A and B cities
        if a_seg[-1] is not city_a: a_seg.reverse()
        if b_seg[0] is not city_b: b_seg.reverse()

        a_seg += b_seg

        # Update endpoints dict
        del self.endpoints[city_a], self.endpoints[city_b]
        self.endpoints[a_seg[0]] = self.endpoints[a_seg[-1]] = a_seg

        return a_seg
    
    def get_route(self) -> list:
        """ Join the various segments into a route that can be drawn

        Returns:
            list: Route of the current cities
        """

        route = []
        seen = set()

        # Combine subsegments if not seen before
        for endpoint in self.endpoints:
            if endpoint not in seen:
                route += self.endpoints[endpoint]
                seen.add(self.endpoints[endpoint][0])
                seen.add(self.endpoints[endpoint][-1])

        return route
    
    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        # Draw MST edges while iterating
        if not self.done:
            draw_edges(window, self.endpoints)

        # Draw full route when complete
        else:
            draw_route(window, self.get_route())

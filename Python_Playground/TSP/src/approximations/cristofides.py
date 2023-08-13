# https://github.com/Retsediv/ChristofidesAlgorithm/blob/master/christofides.py

from itertools import combinations
from collections import defaultdict

from .approximation import Approximation
from .approximation_utils import draw_mst, draw_edge_list, draw_route, calc_route_distance, calc_distance


class Cristofides(Approximation):
    def __init__(self, cities: list) -> None:
        # Step 1 items
        self.num_cities = len(cities)
        self.remaining_cities = cities[:]
        self.mst_done = False
        self.shortest_links = sorted(combinations(cities, 2), key=lambda link: calc_distance(*link))
        self.curr_link = 0
        self.mst = []
        self.mst_edges = {self.first(cities): []}

        # Step 2 items:
        self.vertex_counting_done = False

        # Step 3 items:
        self.min_weight_matching_done = False

        # Step 4 items:
        self.eulerian_done = False

        # Step 5 items:
        self.hamiltonian_done = False
        self.EP = []

        # Step 6 items:
        self.path = []
        self.visited = set()
        self.final_done = False

    def first(self, collection):
        """ Returns first item in a collection

        Args:
            collection (iterable): an iterable to retrieve from

        Returns:
            object: first item in the collection
        """

        return next(iter(collection))

    def run(self) -> tuple[float, bool]:
        """ Runs christofides algorithm
            Step 1: Build MST
            Step 2: Find all cities with an odd degree
            Step 3: Find minimum-weight perfect matching in the subgraph of odd degree nodes
            Step 4: Build eurlerian circuit from the combined edges
            Step 5: Convert to hamiltonian cycle by skipping repeated nodes
            Step 6: Add cities from hamiltonian cycle to final path, ignoring repeats

        Returns:
            tuple[float, bool]: length of the current route and whether the approximation is completed
        """
        # Step 1
        if not self.mst_done:
            self.build_mst()

            if len(self.mst_edges) >= self.num_cities:
                self.convert_mst()
                self.mst_done = True

        # Step 2
        elif not self.vertex_counting_done:
            self.odd_verticies = self.find_odd_degrees()
            self.vertex_counting_done = True

        # Step 3
        elif not self.min_weight_matching_done:
            self.minimum_weight_matching()
            self.min_weight_matching_done = not self.odd_verticies

        # Step 4
        elif not self.eulerian_done:
            self.neighbors = self.get_eulerian_tour()
            self.eulerian_done = True
            self.EP = [self.neighbors[self.mst[0][0]][0]]

        # Step 5
        elif not self.hamiltonian_done:
            self.build_hamiltonian_cycle()
            self.hamiltonian_done = not self.mst
        
        # Step 6
        else:
            self.finalize_route()
            self.final_done = len(self.path) == self.num_cities

        return self.distance, self.complete
    
    def build_mst(self) -> None:
        # Add edge to MST
        a, b = self.first((a, b) for (a, b) in self.shortest_links if (a in self.mst_edges) ^ (b in self.mst_edges))
        if a not in self.mst_edges:
            a, b = b, a

        self.mst_edges[a].append(b)
        self.mst_edges[b] = []

        # Remove from remaining cities for semi-accurate fitness calculation
        if a in self.remaining_cities:
            self.remaining_cities.remove(a)
        if b in self.remaining_cities:
            self.remaining_cities.remove(b)

    def convert_mst(self):
        """ Convert MST into list of edges
        """

        for vertex, connections in self.mst_edges.items():
            for city in connections:
                self.mst.append((vertex, city))

    def find_odd_degrees(self) -> list:
        """ Finds all vertices with an odd degree in the MST

        Returns:
            list: list of odd degree verticies
        """

        degree_counter = defaultdict(lambda: 0)

        for start, stop in self.mst:
            degree_counter[start] += 1
            degree_counter[stop] += 1

        odds = [city for city, count in degree_counter.items() if count % 2 == 1]

        return odds
    
    def minimum_weight_matching(self) -> None:
        """ Add minimum-weight perfect matching of odd degree nodes to MST
        """

        vert = self.odd_verticies.pop()
        
        if self.odd_verticies:
            nearest = min(self.odd_verticies, key=lambda x: calc_distance(vert, x))
        
        if nearest != vert:
            self.mst.append((vert, nearest))
            self.odd_verticies.remove(nearest)

    def get_eulerian_tour(self) -> dict:
        """ Builds eulerian tour

        Returns:
            dict: neighbors of all verticies
        """

        neighbors = defaultdict(lambda: [])
        for city_a, city_b in self.mst:
            neighbors[city_a].append(city_b)
            neighbors[city_b].append(city_a)
        
        return neighbors
    
    def build_hamiltonian_cycle(self) -> None:
        for index, vertex in enumerate(self.EP):
            if self.neighbors[vertex]:
                break

        while self.neighbors[vertex]:
            connected = self.neighbors[vertex].pop()
            self.remove_matched_edge(vertex, connected)
            del self.neighbors[connected][self.neighbors[connected].index(vertex)]

            index += 1
            self.EP.insert(index, connected)
            vertex = connected


    def remove_matched_edge(self, city_a: object, city_b: object) -> None:
        """ Remove edge from edge list if it contains the two provided cities

        Args:
            city_a (City): First city to match
            city_b (City): Second city to match
        """

        if (city_a, city_b) in self.mst:
            self.mst.remove((city_a, city_b))
        elif (city_b, city_a) in self.mst:
            self.mst.remove((city_b, city_a))

    def finalize_route(self) -> None:
        city = self.EP.pop()
        if city not in self.visited:
            self.visited.add(city)
            self.path.append(city)
    
    @property
    def distance(self) -> float:
        if not self.mst_done:
            links = [(parent, child) for parent in self.mst_edges for child in self.mst_edges[parent]]
            if len(self.remaining_cities) > 1:
                return calc_route_distance(self.remaining_cities) + sum(calc_distance(p, c) for p, c in links)
            else:
                return sum(calc_distance(p, c) for p, c in links)
            
        elif not self.hamiltonian_done:
            links = [(start, end) for start, end in self.mst]
            return sum(calc_distance(start, end) for start, end in links) + calc_route_distance(self.EP)
        
        else:
            return calc_route_distance(self.EP) + calc_route_distance(self.path)
    
    @property
    def complete(self) -> bool:
        """ Outputs whether the algorithm is complete or not

        Returns:
            bool: Whether algorithm is complete
        """

        return self.mst_done and self.vertex_counting_done and self.min_weight_matching_done and self.eulerian_done and self.hamiltonian_done and self.final_done
    
    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """

        if not self.mst_done:
            draw_mst(window, self.mst_edges)

        elif not self.hamiltonian_done:
            draw_edge_list(window, self.mst)
            draw_route(window, self.EP)
        
        else:
            draw_route(window, self.EP)
            draw_route(window, self.path)
        
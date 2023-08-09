import itertools

from .approximation import Approximation
from .approximation_utils import draw_route, calc_fitness_memo


class DivideAndConquer(Approximation):
    def __init__(self, cities: list, min_segment_size: int=5) -> None:
        self.min_segment_size = min_segment_size
        self.curr_segments = [cities]
        self.new_segments = []
        self.done_split = False

    def run(self) -> tuple[float, bool]:
        """ Divide list of cities into smaller subsegments. Then, combine subsegments in an optimal manner

        Returns:
            tuple[float, bool]: The score of the current route and whether the approximation is completed
        """

        # Divide
        if not self.done_split:
            self.split_cities()
            self.curr_segments = self.new_segments
            self.new_segments = []
            if len(self.curr_segments[0]) <= self.min_segment_size:
                self.done_split = True

        # Reset
        elif len(self.curr_segments) <= 1:
            self.curr_segments = self.curr_segments + self.new_segments
            self.new_segments = []
        
        # Conquer
        else:
            new_segment = self.join_tours(*self.curr_segments[:2])
            self.new_segments.append(new_segment)
            self.curr_segments = self.curr_segments[2:]

        return calc_fitness_memo(list(itertools.chain.from_iterable(self.curr_segments)) + list(itertools.chain.from_iterable(self.new_segments))), len(self.curr_segments) == 1 and len(self.new_segments) == 0
    
    def split_cities(self) -> None:
        """ Divide city list into smaller subsegments
        """

        for segment in self.curr_segments:
            coord = self._X if self.extent(segment, self._X) > self.extent(segment, self._Y) else self._Y
            cities = sorted(segment, key=coord)
            middle = len(cities) // 2
            self.new_segments.append(cities[:middle])
            self.new_segments.append(cities[middle:])

    def _X(self, city) -> int:
        """ Get X coord from city

        Args:
            city (City): City object

        Returns:
            int: X coord of city object
        """

        return city.x
    
    def _Y(self, city) -> int:
        """ Get Y coord from city

        Args:
            city (City): City object

        Returns:
            int: Y coord of city object
        """

        return city.y
    
    def extent(self, cities: list, coord) -> int:
        """ Determine the range of a give coordinate in a segment of cities

        Args:
            cities (list): Subsegment to evaluate
            coord (function): Either _X or _Y, to retrieve coordinates

        Returns:
            int: Range of the given coordinate accross the segment
        """
        return max([coord(city) for city in cities]) - min([coord(city) for city in cities])

    def brute_force(self, segment: list) -> list:
        """ Optimizes a given segment via the brute force TSP algorithm

        Args:
            segment (list): Subsegment to find solution for

        Returns:
            list: Optimzed route of the given segment
        """

        start, *others = segment
        return self.shortest([[start, *perm] for perm in itertools.permutations(others)])
    
    def join_tours(self, tour1: list, tour2: list) -> list:
        """ Joins two tours in the optimal manner

        Args:
            tour1 (list): First segment
            tour2 (list): Second segment

        Returns:
            list: Combined tour
        """
        segments1, segments2 = self.rotations(tour1), self.rotations(tour2)
        return self.shortest(s1 + s3
                    for s1 in segments1
                    for s2 in segments2
                    for s3 in (s2, s2[::-1]))

    def rotations(self, sequence: list) -> list[list]:
        """ Generates all possible rotations of a given sequence

        Args:
            sequence (list): Segment to rotate

        Returns:
            list[list]: list of all possible rotations
        """
        return [sequence[i:] + sequence[:i] for i in range(len(sequence))]
    
    def shortest(self, tours: list[list]) -> list:
        """ Identifies the best tour from a list of tours

        Args:
            tours (list[list]): List of tours

        Returns:
            list: The best tour in the list of tours
        """

        return max(tours, key=calc_fitness_memo)
    
    def draw(self, window) -> None:
        """ Draw calculated route

        Args:
            window (pygame.surface.Surface): Game window to draw onto
        """
        
        for segment in self.curr_segments:
            draw_route(window, segment)
        for segment in self.new_segments:
            draw_route(window, segment)

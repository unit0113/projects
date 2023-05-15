from graph import Graph
from weighted_graph import WeightedGraph
import unittest


class TestGraphCreation(unittest.TestCase):
    def runTest(self):
        g = Graph()
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_vertex('C')
        answer = {'A': set(), 'B': set(), 'C': set()}
        self.assertEqual(g.adj_list, answer, 'Error during insertion')

        g.add_directed_edge('A', 'B')
        answer['A'].add('B')
        self.assertEqual(g.adj_list, answer, 'Error adding directed edge')

        g.add_undirected_edge('B', 'C')
        answer['B'].add('C')
        answer['C'].add('B')
        self.assertEqual(g.adj_list, answer, 'Error adding undirected edge')

        with self.assertRaises(ValueError):
            g.add_directed_edge('B', 'D')

        with self.assertRaises(ValueError):
            g.add_undirected_edge('B', 'D')


class TestTopologicalSearch(unittest.TestCase):
    def runTest(self):
        g = Graph()
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_vertex('C')
        g.add_directed_edge('A', 'B')
        g.add_directed_edge('B', 'C')
        answer = ['A', 'B', 'C']
        self.assertTrue(g._valid_topological_order(g.topological_sort()), 'Incorrect topological ordering; small test')

        g = Graph()
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_vertex('C')
        g.add_vertex('D')
        g.add_vertex('E')
        g.add_vertex('F')
        g.add_vertex('G')
        g.add_vertex('H')
        g.add_vertex('I')
        g.add_vertex('J')
        g.add_vertex('K')
        g.add_directed_edge('A', 'B')
        g.add_directed_edge('B', 'E')
        g.add_directed_edge('E', 'H')
        g.add_directed_edge('H', 'J')
        g.add_directed_edge('A', 'D')
        g.add_directed_edge('D', 'E')
        g.add_directed_edge('D', 'F')
        g.add_directed_edge('C', 'F')
        g.add_directed_edge('E', 'G')
        g.add_directed_edge('F', 'G')
        g.add_directed_edge('F', 'I')
        g.add_directed_edge('G', 'I')
        g.add_directed_edge('G', 'J')
        self.assertTrue(g._valid_topological_order(g.topological_sort()), 'Incorrect topological ordering; large test')


class TestTopologicalSearchKahn(unittest.TestCase):
    def runTest(self):
        g = Graph()
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_vertex('C')
        g.add_directed_edge('A', 'B')
        g.add_directed_edge('B', 'C')
        answer = ['A', 'B', 'C']
        self.assertEqual(g.topological_sort_kahn(), answer, 'Incorrect topological ordering; small test')

        g = Graph()
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_vertex('C')
        g.add_vertex('D')
        g.add_vertex('E')
        g.add_vertex('F')
        g.add_vertex('G')
        g.add_vertex('H')
        g.add_vertex('I')
        g.add_vertex('J')
        g.add_vertex('K')
        g.add_directed_edge('A', 'B')
        g.add_directed_edge('B', 'E')
        g.add_directed_edge('E', 'H')
        g.add_directed_edge('H', 'J')
        g.add_directed_edge('A', 'D')
        g.add_directed_edge('D', 'E')
        g.add_directed_edge('D', 'F')
        g.add_directed_edge('C', 'F')
        g.add_directed_edge('E', 'G')
        g.add_directed_edge('F', 'G')
        g.add_directed_edge('F', 'I')
        g.add_directed_edge('G', 'I')
        g.add_directed_edge('G', 'J')
        self.assertTrue(g._valid_topological_order(g.topological_sort_kahn()), 'Incorrect topological ordering; large test')


class TestMaximalStronglyConnectedComponents(unittest.TestCase):
    def runTest(self):
        g = Graph()
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_vertex('C')
        g.add_directed_edge('A', 'B')
        g.add_directed_edge('B', 'C')
        g.add_directed_edge('C', 'A')
        self.assertEqual(g.get_msccs(), [['A', 'B', 'C']], 'Incorrect MSCC; small test')

        g1 = Graph()
        g1.add_vertex(0)
        g1.add_vertex(1)
        g1.add_vertex(2)
        g1.add_vertex(3)
        g1.add_vertex(4)
        g1.add_vertex(5)
        g1.add_vertex(6)
        g1.add_vertex(7)

        g1.add_directed_edge(0, 1)
        g1.add_directed_edge(1, 2)
        g1.add_directed_edge(2, 3)
        g1.add_directed_edge(3, 0)
        g1.add_directed_edge(2, 4)
        g1.add_directed_edge(4, 5)
        g1.add_directed_edge(5, 6)
        g1.add_directed_edge(6, 4)
        g1.add_directed_edge(6, 7)
        self.assertEqual(g1.get_msccs(), [[0, 1, 2, 3], [4, 5, 6], [7]], 'Incorrect MSCC; large test')


class TestMinimumSpanningTree(unittest.TestCase):
    def runTest(self):
        g = WeightedGraph()
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_vertex('C')
        g.add_vertex('D')
        g.add_directed_edge('A', 'B', 0.5)
        g.add_directed_edge('B', 'C', 1)
        g.add_directed_edge('A', 'D', 2)
        g.add_directed_edge('C', 'D', 5)
        self.assertEqual(g.get_min_spanning_tree()[1], 3.5, 'Incorrect Minimum Spanning Tree; small test')

        g = WeightedGraph()
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_vertex('C')
        g.add_vertex('D')
        g.add_vertex('E')
        g.add_vertex('F')
        g.add_vertex('G')
        g.add_vertex('H')
        g.add_vertex('I')

        g.add_directed_edge('A', 'B', 4)
        g.add_directed_edge('B', 'C', 8)
        g.add_directed_edge('C', 'D', 7)
        g.add_directed_edge('D', 'E', 9)
        g.add_directed_edge('E', 'F', 10)
        g.add_directed_edge('F', 'G', 2)
        g.add_directed_edge('G', 'H', 1)
        g.add_directed_edge('A', 'H', 8)
        g.add_directed_edge('B', 'H', 11)
        g.add_directed_edge('H', 'I', 7)
        g.add_directed_edge('G', 'I', 6)
        g.add_directed_edge('C', 'I', 2)
        g.add_directed_edge('C', 'F', 4)
        g.add_directed_edge('D', 'F', 14)
        self.assertEqual(g.get_min_spanning_tree()[1], 37, 'Incorrect Minimum Spanning Tree; large test')


class TestSingleSourceShortestPathBellmanFord(unittest.TestCase):
    def runTest(self):
        g = WeightedGraph()
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_vertex('C')
        g.add_vertex('D')
        g.add_vertex('E')
        g.add_directed_edge('A', 'B', -1)
        g.add_directed_edge('A', 'C', 4)
        g.add_directed_edge('B', 'C', 3)
        g.add_directed_edge('B', 'E', 2)
        g.add_directed_edge('E', 'D', -3)
        g.add_directed_edge('B', 'D', 2)
        g.add_directed_edge('D', 'B', 1)
        g.add_directed_edge('D', 'C', 5)
        answer = {'A': 0, 'B': -1, 'C': 2, 'D': -2, 'E': 1}
        self.assertEqual(g.get_sssp_bf('A'), answer, 'Incorrect Bellman-Ford single source shortest path')

        g.add_directed_edge('C', 'A', -5)
        with self.assertRaises(ValueError):
            g.get_sssp_bf('A')


class TestSingleSourceShortestPathDijkstra(unittest.TestCase):
    def runTest(self):
        g = WeightedGraph()
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_vertex('C')
        g.add_vertex('D')
        g.add_vertex('E')
        g.add_vertex('F')
        g.add_vertex('G')
        g.add_vertex('H')
        g.add_vertex('I')

        g.add_undirected_edge('A', 'B', 4)
        g.add_undirected_edge('B', 'C', 8)
        g.add_undirected_edge('C', 'D', 7)
        g.add_undirected_edge('D', 'E', 9)
        g.add_undirected_edge('E', 'F', 10)
        g.add_undirected_edge('F', 'G', 2)
        g.add_undirected_edge('G', 'H', 1)
        g.add_undirected_edge('A', 'H', 8)
        g.add_undirected_edge('B', 'H', 11)
        g.add_undirected_edge('H', 'I', 7)
        g.add_undirected_edge('G', 'I', 6)
        g.add_undirected_edge('C', 'I', 2)
        g.add_undirected_edge('C', 'F', 4)
        g.add_undirected_edge('D', 'F', 14)
        answer = {'A': 0, 'B': 4, 'C': 12, 'D': 19, 'E': 21, 'F': 11, 'G': 9, 'H': 8, 'I': 14}
        self.assertEqual(g.get_sssp_dijkstra('A')[0], answer, 'Incorrect Dijkstra single source shortest path')

        self.assertEqual(g.get_shortest_path_dijkstra('A', 'E'), (['A', 'H', 'G', 'F', 'E'], 21), 'Incorrect shortest path via Dijkstra')


class TestAllPairsShortestPathFloydWarshall(unittest.TestCase):
    def runTest(self):
        g = WeightedGraph()
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_vertex('C')
        g.add_vertex('D')
        g.add_directed_edge('A', 'B', 3)
        g.add_directed_edge('A', 'D', 5)
        g.add_directed_edge('B', 'A', 2)
        g.add_directed_edge('B', 'D', 4)
        g.add_directed_edge('C', 'B', 1)
        g.add_directed_edge('D', 'C', 2)
        answer = [[0, 3, 7, 5], [2, 0, 6, 4], [3, 1, 0, 5], [5, 3, 2, 0]]
        self.assertEqual(g.get_apsp_fw(), answer, 'Incorrect Floyd Warshall all pairs shortest path')










unittest.main()

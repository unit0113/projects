from graph import Graph
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


class TestGraphCreation(unittest.TestCase):
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








unittest.main()

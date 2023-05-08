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
        self.assertEqual(g.topological_sort(), answer, 'Incorrect topological ordering; small test')

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



unittest.main()
import unittest
from random import shuffle
from program3 import program3
from program4 import program4


# python -m unittest tests


class TestProgram1(unittest.TestCase):
    def test_3(self):
        n = 7
        w = 10
        heights = [21, 19, 17, 16, 11, 5, 1]
        widths = [7, 1, 2, 3, 5, 8, 1]

        m, total_height, num_statues = program3(n, w, heights, widths)
        self.assertEqual(m, 3)
        self.assertEqual(total_height, 42)
        self.assertEqual(len(num_statues), m)

        print(m)
        print(total_height)
        print(num_statues)
        print()

    def test_3_2(self):
        n = 7
        w = 10
        heights = [12, 10, 9, 7, 8, 10, 11]
        widths = [3, 2, 3, 4, 3, 2, 3]

        m, total_height, num_statues = program3(n, w, heights, widths)
        self.assertEqual(m, 3)
        self.assertEqual(total_height, 30)
        self.assertEqual(len(num_statues), m)

        print(m)
        print(total_height)
        print(num_statues)
        print()

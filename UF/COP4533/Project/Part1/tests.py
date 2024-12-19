import unittest
from program1 import program1
from program2 import program2

# python -m unittest tests


class TestProgram1(unittest.TestCase):
    def test_1(self):
        m, total_height, num_statues = program1(
            7, 10, [21, 19, 17, 16, 11, 5, 1], [7, 1, 2, 3, 5, 8, 1]
        )
        self.assertEqual(m, 3)
        self.assertEqual(total_height, 42)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues[0], 3)
        self.assertEqual(num_statues[1], 2)
        self.assertEqual(num_statues[2], 2)

        print(m)
        print(total_height)
        print(num_statues)
        print()


class TestProgram2(unittest.TestCase):
    def test_1(self):
        m, total_height, num_statues = program2(
            7, 10, [21, 19, 17, 16, 11, 5, 1], [7, 1, 2, 3, 5, 8, 1]
        )
        self.assertEqual(m, 3)
        self.assertEqual(total_height, 42)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues[0], 3)
        self.assertEqual(num_statues[1], 2)
        self.assertEqual(num_statues[2], 2)

        print(m)
        print(total_height)
        print(num_statues)
        print()

    def test_3(self):
        m, total_height, num_statues = program2(
            7, 10, [12, 10, 9, 7, 8, 10, 11], [3, 2, 3, 4, 3, 2, 3]
        )
        self.assertEqual(m, 3)
        self.assertEqual(total_height, 30)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues[0], 3)
        self.assertEqual(num_statues[1], 1)
        self.assertEqual(num_statues[2], 3)
        print(m)
        print(total_height)
        print(num_statues)
        print()

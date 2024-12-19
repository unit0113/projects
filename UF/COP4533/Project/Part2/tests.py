import unittest
from program3 import program3
from program4 import program4
from program5A import program5A
from program5B import program5B


# python -m unittest tests


class TestProgram3(unittest.TestCase):
    def test_3_1(self):
        n = 7
        w = 10
        heights = [21, 19, 17, 16, 11, 5, 1]
        widths = [7, 1, 2, 3, 5, 8, 1]

        m, total_height, num_statues = program3(n, w, heights, widths)
        self.assertEqual(m, 3)
        self.assertEqual(total_height, 42)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [3, 2, 2])

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
        self.assertEqual(num_statues, [3, 1, 3])

        print(m)
        print(total_height)
        print(num_statues)
        print()

    def test_3_3(self):
        n = 10
        w = 15
        heights = [31, 29, 25, 19, 4, 2, 4, 7, 8, 9]
        widths = [3, 4, 6, 5, 3, 1, 2, 14, 1, 3]

        m, total_height, num_statues = program3(n, w, heights, widths)
        self.assertEqual(m, 4)
        self.assertEqual(total_height, 66)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [3, 4, 1, 2])

        print(m)
        print(total_height)
        print(num_statues)
        print()

    def test_3_4(self):
        n = 20
        w = 25
        heights = [
            65,
            41,
            31,
            21,
            11,
            1,
            2,
            3,
            3,
            5,
            6,
            7,
            8,
            9,
            13,
            15,
            17,
            18,
            19,
            100,
        ]
        widths = [20, 2, 6, 12, 8, 4, 6, 7, 8, 1, 6, 7, 9, 8, 7, 8, 12, 16, 20, 5]

        m, total_height, num_statues = program3(n, w, heights, widths)
        self.assertEqual(m, 8)
        self.assertEqual(total_height, 262)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [2, 2, 4, 4, 3, 2, 1, 2])

        print(m)
        print(total_height)
        print(num_statues)
        print()


class TestProgram4(unittest.TestCase):
    def test_4_1(self):
        n = 7
        w = 10
        heights = [21, 19, 17, 16, 11, 5, 1]
        widths = [7, 1, 2, 3, 5, 8, 1]

        m, total_height, num_statues = program4(n, w, heights, widths)
        self.assertEqual(m, 3)
        self.assertEqual(total_height, 42)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [3, 2, 2])

        print(m)
        print(total_height)
        print(num_statues)
        print()

    def test_4_2(self):
        n = 7
        w = 10
        heights = [12, 10, 9, 7, 8, 10, 11]
        widths = [3, 2, 3, 4, 3, 2, 3]

        m, total_height, num_statues = program4(n, w, heights, widths)
        self.assertEqual(m, 3)
        self.assertEqual(total_height, 30)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [3, 1, 3])

        print(m)
        print(total_height)
        print(num_statues)
        print()

    def test_4_3(self):
        n = 10
        w = 15
        heights = [31, 29, 25, 19, 4, 2, 4, 7, 8, 9]
        widths = [3, 4, 6, 5, 3, 1, 2, 14, 1, 3]

        m, total_height, num_statues = program4(n, w, heights, widths)
        self.assertEqual(m, 4)
        self.assertEqual(total_height, 66)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [3, 4, 1, 2])

        print(m)
        print(total_height)
        print(num_statues)
        print()

    def test_4_4(self):
        n = 20
        w = 25
        heights = [
            65,
            41,
            31,
            21,
            11,
            1,
            2,
            3,
            3,
            5,
            6,
            7,
            8,
            9,
            13,
            15,
            17,
            18,
            19,
            100,
        ]
        widths = [20, 2, 6, 12, 8, 4, 6, 7, 8, 1, 6, 7, 9, 8, 7, 8, 12, 16, 20, 5]

        m, total_height, num_statues = program4(n, w, heights, widths)
        self.assertEqual(m, 8)
        self.assertEqual(total_height, 262)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [2, 2, 4, 4, 3, 2, 1, 2])

        print(m)
        print(total_height)
        print(num_statues)
        print()


class TestProgram5A(unittest.TestCase):
    def test_5A_1(self):
        n = 7
        w = 10
        heights = [21, 19, 17, 16, 11, 5, 1]
        widths = [7, 1, 2, 3, 5, 8, 1]

        m, total_height, num_statues = program5A(n, w, heights, widths)
        self.assertEqual(m, 3)
        self.assertEqual(total_height, 42)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [3, 2, 2])

        print(m)
        print(total_height)
        print(num_statues)
        print()

    def test_5A_2(self):
        n = 7
        w = 10
        heights = [12, 10, 9, 7, 8, 10, 11]
        widths = [3, 2, 3, 4, 3, 2, 3]

        m, total_height, num_statues = program5A(n, w, heights, widths)
        self.assertEqual(m, 3)
        self.assertEqual(total_height, 30)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [3, 1, 3])

        print(m)
        print(total_height)
        print(num_statues)
        print()

    def test_5A_3(self):
        n = 10
        w = 15
        heights = [31, 29, 25, 19, 4, 2, 4, 7, 8, 9]
        widths = [3, 4, 6, 5, 3, 1, 2, 14, 1, 3]

        m, total_height, num_statues = program5A(n, w, heights, widths)
        self.assertEqual(m, 4)
        self.assertEqual(total_height, 66)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [3, 4, 1, 2])

        print(m)
        print(total_height)
        print(num_statues)
        print()

    def test_5A_4(self):
        n = 20
        w = 25
        heights = [
            65,
            41,
            31,
            21,
            11,
            1,
            2,
            3,
            3,
            5,
            6,
            7,
            8,
            9,
            13,
            15,
            17,
            18,
            19,
            100,
        ]
        widths = [20, 2, 6, 12, 8, 4, 6, 7, 8, 1, 6, 7, 9, 8, 7, 8, 12, 16, 20, 5]

        m, total_height, num_statues = program5A(n, w, heights, widths)
        self.assertEqual(m, 8)
        self.assertEqual(total_height, 262)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [2, 2, 4, 4, 3, 2, 1, 2])

        print(m)
        print(total_height)
        print(num_statues)
        print()


class TestProgram5B(unittest.TestCase):
    def test_5B_1(self):
        n = 7
        w = 10
        heights = [21, 19, 17, 16, 11, 5, 1]
        widths = [7, 1, 2, 3, 5, 8, 1]

        m, total_height, num_statues = program5B(n, w, heights, widths)
        self.assertEqual(m, 3)
        self.assertEqual(total_height, 42)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [3, 2, 2])

        print(m)
        print(total_height)
        print(num_statues)
        print()

    def test_5B_2(self):
        n = 7
        w = 10
        heights = [12, 10, 9, 7, 8, 10, 11]
        widths = [3, 2, 3, 4, 3, 2, 3]

        m, total_height, num_statues = program5B(n, w, heights, widths)
        self.assertEqual(m, 3)
        self.assertEqual(total_height, 30)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [3, 1, 3])

        print(m)
        print(total_height)
        print(num_statues)
        print()

    def test_5B_3(self):
        n = 10
        w = 15
        heights = [31, 29, 25, 19, 4, 2, 4, 7, 8, 9]
        widths = [3, 4, 6, 5, 3, 1, 2, 14, 1, 3]

        m, total_height, num_statues = program5B(n, w, heights, widths)
        self.assertEqual(m, 4)
        self.assertEqual(total_height, 66)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [3, 4, 1, 2])

        print(m)
        print(total_height)
        print(num_statues)
        print()

    def test_5B_4(self):
        n = 20
        w = 25
        heights = [
            65,
            41,
            31,
            21,
            11,
            1,
            2,
            3,
            3,
            5,
            6,
            7,
            8,
            9,
            13,
            15,
            17,
            18,
            19,
            100,
        ]
        widths = [20, 2, 6, 12, 8, 4, 6, 7, 8, 1, 6, 7, 9, 8, 7, 8, 12, 16, 20, 5]

        m, total_height, num_statues = program5B(n, w, heights, widths)
        self.assertEqual(m, 8)
        self.assertEqual(total_height, 262)
        self.assertEqual(len(num_statues), m)
        self.assertEqual(num_statues, [2, 2, 4, 4, 3, 2, 1, 2])

        print(m)
        print(total_height)
        print(num_statues)
        print()

import unittest
import linear_search as ls
import binary_search as bs
import jump_search as js
import interpolation_search as ints


class TestLinkedList(unittest.TestCase):
    
    def test_linear_search(self):
        arr = [1,2,3,4,5,6,7,8,9,10]
        self.assertEqual(True, ls.linear_search(arr, 5))
        self.assertEqual(False, ls.linear_search(arr, 20))
        arr = []
        self.assertEqual(False, ls.linear_search(arr, 5))

    def test_binary_search(self):
        arr = [1,2,3,4,5,6,7,8,9,10]
        self.assertEqual(True, bs.binary_search(arr, 5))
        self.assertEqual(False, bs.binary_search(arr, 20))
        arr = []
        self.assertEqual(False, bs.binary_search(arr, 5))

    def test_jump_search(self):
        arr = [1,2,3,4,5,6,7,8,9,10]
        self.assertEqual(True, js.jump_search(arr, 5))
        self.assertEqual(False, js.jump_search(arr, 20))
        arr = []
        self.assertEqual(False, js.jump_search(arr, 5))

    def test_interpolation_search(self):
        arr = [1,2,3,4,5,6,7,8,9,10]
        self.assertEqual(True, ints.interpolation_search(arr, 5))
        self.assertEqual(False, ints.interpolation_search(arr, 20))
        arr = []
        self.assertEqual(False, ints.interpolation_search(arr, 5))

if __name__ == '__main__':
    unittest.main()
import unittest
import avl as avl

class TestLinkedList(unittest.TestCase):
    
    def test_add_front_single(self):
        a = ll.Linked_List(5)
        a.add_front(10)
        self.assertEqual(10, a.head.value)
        self.assertEqual(5, a.tail.value)
        self.assertEqual(2, a.length)
        self.assertEqual(5, a.head.next.value)
        self.assertEqual(None, a.head.prev)
        self.assertEqual(10, a.tail.prev.value)
        with self.assertRaises(ValueError):
            a.add_front()


if __name__ == '__main__':
    unittest.main()
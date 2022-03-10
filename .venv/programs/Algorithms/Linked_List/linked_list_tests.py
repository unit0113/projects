import unittest
import linked_list as ll

class TestLinkedList(unittest.TestCase):

    def test_node_creation(self):
        a = ll.Node(5)
        self.assertEqual(5, a.value)
        self.assertEqual(None, a.next)
        self.assertEqual(None, a.prev)

    def test_linked_list_creation_empty(self):
        a = ll.Linked_List()
        self.assertEqual(None, a.head)
        self.assertEqual(None, a.tail)
        self.assertEqual(0, a.length)

    def test_linked_list_creation_single(self):
        a = ll.Linked_List(5)
        self.assertEqual(5, a.head.value)
        self.assertEqual(5, a.tail.value)
        self.assertEqual(1, a.length)
        self.assertEqual(None, a.head.next)
        self.assertEqual(None, a.head.prev)

    def test_linked_list_creation_multiple(self):
        a = ll.Linked_List(1, 2, 3, 4, 5)
        self.assertEqual(5, a.head.value)
        self.assertEqual(1, a.tail.value)
        self.assertEqual(5, a.length)
        self.assertEqual(4, a.head.next.value)
        self.assertEqual(None, a.head.prev)
        self.assertEqual(2, a.tail.prev.value)

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

    def test_add_front_multiple(self):
        a = ll.Linked_List(5)
        a.add_front(10, 20, 30)
        self.assertEqual(30, a.head.value)
        self.assertEqual(5, a.tail.value)
        self.assertEqual(4, a.length)
        self.assertEqual(20, a.head.next.value)
        self.assertEqual(None, a.head.prev)
        self.assertEqual(10, a.tail.prev.value)    

    def test_add_rear_single(self):
        a = ll.Linked_List(10)
        a.add_rear(5)
        self.assertEqual(10, a.head.value)
        self.assertEqual(5, a.tail.value)
        self.assertEqual(2, a.length)
        self.assertEqual(5, a.head.next.value)
        self.assertEqual(None, a.head.prev)
        self.assertEqual(10, a.tail.prev.value)
        with self.assertRaises(ValueError):
            a.add_rear()

    def test_add_rear_multiple(self):
        a = ll.Linked_List(10)
        a.add_rear(5, 4, 3)
        self.assertEqual(10, a.head.value)
        self.assertEqual(3, a.tail.value)
        self.assertEqual(4, a.length)
        self.assertEqual(5, a.head.next.value)
        self.assertEqual(None, a.head.prev)
        self.assertEqual(4, a.tail.prev.value)

    def test_remove_front(self):
        a = ll.Linked_List()
        with self.assertRaises(ValueError):
            a.remove_first()
        a.add_front(5)
        self.assertEqual(5, a.remove_first())
        self.assertEqual(None, a.head)
        self.assertEqual(None, a.tail)
        with self.assertRaises(ValueError):
            a.remove_first()
        a.add_front(1, 2, 3, 4, 5)
        self.assertEqual(5, a.remove_first())
        self.assertEqual(4, a.head.value)
        self.assertEqual(None, a.head.prev)
        self.assertEqual(4, a.length)

    def test_remove_rear(self):
        a = ll.Linked_List()
        with self.assertRaises(ValueError):
            a.remove_rear()
        a.add_rear(5)
        self.assertEqual(5, a.remove_rear())
        self.assertEqual(None, a.head)
        self.assertEqual(None, a.tail)
        with self.assertRaises(ValueError):
            a.remove_first()
        a.add_front(1, 2, 3, 4, 5)
        self.assertEqual(5, a.remove_first())
        self.assertEqual(4, a.head.value)
        self.assertEqual(None, a.head.prev)
        self.assertEqual(4, a.length)


if __name__ == '__main__':
    unittest.main()
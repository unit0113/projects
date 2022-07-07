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

    def test_remove_last(self):
        a = ll.Linked_List()
        with self.assertRaises(ValueError):
            a.remove_last()
        a.add_rear(5)
        self.assertEqual(5, a.remove_last())
        self.assertEqual(None, a.head)
        self.assertEqual(None, a.tail)
        with self.assertRaises(ValueError):
            a.remove_last()
        a.add_front(1, 2, 3, 4, 5)
        self.assertEqual(1, a.remove_last())
        self.assertEqual(5, a.head.value)
        self.assertEqual(None, a.head.prev)
        self.assertEqual(4, a.length)

    def test_find(self):
        a = ll.Linked_List(1, 2, 3, 4, 5)
        self.assertEqual(True, a.find(5))
        self.assertEqual(True, a.find(1))
        self.assertEqual(False, a.find(6))
        a = ll.Linked_List()
        self.assertEqual(False, a.find(6))

    def test_remove(self):
        a = ll.Linked_List(1, 2, 3, 4, 5)
        a.remove(1)
        self.assertEqual(2, a.tail.value)
        a.remove(5)
        self.assertEqual(4, a.head.value)
        a.remove(3)
        self.assertEqual(2, a.head.next.value)
        self.assertEqual(4, a.tail.prev.value)

    def test_peek(self):
        a = ll.Linked_List(1, 2, 3, 4, 5)
        self.assertEqual(5, a.peek())

    def test_peek_tail(self):
        a = ll.Linked_List(1, 2, 3, 4, 5)
        self.assertEqual(1, a.peek_tail())

    def test_iterator(self):
        a = ll.Linked_List(1, 2, 3, 4, 5)
        arr = []
        for item in a:
            arr.append(item)
        self.assertEqual([5, 4, 3, 2, 1], arr)
        a = ll.Linked_List()
        arr = []
        for item in a:
            arr.append(item)
        self.assertEqual([], arr)     

    def test_contains(self):
        a = ll.Linked_List(1, 2, 3, 4, 5)
        self.assertEqual(True, 5 in a)
        self.assertEqual(True, 6 not in a)
        self.assertEqual(False, 6 in a)
        self.assertEqual(True, 6 not in a)
        a = ll.Linked_List()
        self.assertEqual(False, 5 in a)
        self.assertEqual(True, 5 not in a)


if __name__ == '__main__':
    unittest.main()
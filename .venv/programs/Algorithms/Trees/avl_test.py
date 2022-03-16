import unittest
import avl as avl

class TestLinkedList(unittest.TestCase):

    def test_create_node(self):
        a = avl.Node(5, 10)
        self.assertEqual(10, a.value)
        self.assertEqual(5, a.key)
        self.assertEqual(None, a.parent)
        self.assertEqual(None, a.left)
        self.assertEqual(None, a.right)
        self.assertEqual(1, a.height)

    def test_left_rotate(self):
        a = avl.AVL_Tree()
        b = avl.Node(5, 5)
        c = avl.Node(10, 10)
        d = avl.Node(20,20)
        a.root = b
        b.right = c
        c.parent = b
        c.right = d
        d.parent = c
        a.__left_rotation__(b)
        self.assertEqual(a.root, c)
        self.assertEqual(c.left, b)
        self.assertEqual(c.right, d)
        self.assertEqual(b.parent, c)
        self.assertEqual(d.parent, c)
        e = avl.Node(30,30)
        f = avl.Node(40,40)
        g = avl.Node(25,25)
        d.right = e
        e.parent = d
        e.right = f
        e.left = g
        g.parent = e
        f.parent = e
        a.__left_rotation__(d)
        self.assertEqual(a.root, c)
        self.assertEqual(d.parent, e)
        self.assertEqual(e.left, d)
        self.assertEqual(e.parent, c)
        self.assertEqual(e.right, f)
        self.assertEqual(f.parent, e)
        self.assertEqual(g.parent, d)
        self.assertEqual(d.right, g)

    def test_right_rotate(self):
        a = avl.AVL_Tree()
        b = avl.Node(20, 20)
        c = avl.Node(10, 10)
        d = avl.Node(5, 5)
        a.root = b
        b.left = c
        c.parent = b
        c.left = d
        d.parent = c
        a.__right_rotation__(b)
        self.assertEqual(a.root, c)
        self.assertEqual(c.right, b)
        self.assertEqual(c.left, d)
        self.assertEqual(b.parent, c)
        self.assertEqual(d.parent, c)
        e = avl.Node(3,3)
        f = avl.Node(2,2)
        g = avl.Node(4,4)
        d.left = e
        e.parent = d
        e.left = f
        e.right = g
        g.parent = e
        f.parent = e
        a.__right_rotation__(d)
        self.assertEqual(a.root, c)
        self.assertEqual(d.parent, e)
        self.assertEqual(e.right, d)
        self.assertEqual(e.parent, c)
        self.assertEqual(e.left, f)
        self.assertEqual(f.parent, e)
        self.assertEqual(g.parent, d)
        self.assertEqual(d.left, g)

    
    '''def test_add_front_single(self):
        a = ll.Linked_List(5)
        a.add_front(10)
        self.assertEqual(10, a.head.value)
        self.assertEqual(5, a.tail.value)
        self.assertEqual(2, a.length)
        self.assertEqual(5, a.head.next.value)
        self.assertEqual(None, a.head.prev)
        self.assertEqual(10, a.tail.prev.value)
        with self.assertRaises(ValueError):
            a.add_front()'''


if __name__ == '__main__':
    unittest.main()
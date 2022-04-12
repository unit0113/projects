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
        a._left_rotation(b)
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
        a._left_rotation(d)
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
        a._right_rotation(b)
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
        a._right_rotation(d)
        self.assertEqual(a.root, c)
        self.assertEqual(d.parent, e)
        self.assertEqual(e.right, d)
        self.assertEqual(e.parent, c)
        self.assertEqual(e.left, f)
        self.assertEqual(f.parent, e)
        self.assertEqual(g.parent, d)
        self.assertEqual(d.left, g)


    def test_insert(self):
        a = avl.AVL_Tree()
        a[5] = 5
        self.assertEqual(a.root.value, 5)
        a[7] = 7
        a[9] = 9
        self.assertEqual(a.root.value, 7)
        self.assertEqual(a.root.height, 2)
        self.assertEqual(a.root.left.height, 1)
        a[8] = 8
        self.assertEqual(a.root.height, 3)
        self.assertEqual(a.root.right.height, 2)
        self.assertEqual(a.root.right.left.height, 1)
        self.assertEqual(a.root.right.left.value, 8)
        a[10] = 10
        a[11] = 11
        self.assertEqual(a.root.value, 9)
        self.assertEqual(a.root.left.value, 7)
        self.assertEqual(a.root.left.right.value, 8)
        a[9] = 20
        self.assertEqual(a.root.value, 20)

    def test_get_item(self):
        a = avl.AVL_Tree()
        a[5] = 5
        self.assertEqual(a[5], 5)
        a[5] = 10
        self.assertEqual(a[5], 10)

    def test_successor(self):
        a = avl.AVL_Tree()
        a[5] = 5
        a[7] = 7
        a[9] = 9
        a[8] = 8
        a[10] = 10
        a[11] = 11
        a[9] = 20
        self.assertEqual(a.successor(10).key, 11)
        self.assertEqual(a.successor(7).key, 8)

    def test_delete(self):
        a = avl.AVL_Tree()
        a[5] = 5
        a[7] = 7
        a[9] = 9
        a[8] = 8
        a[10] = 10
        a[11] = 11
        a.delete(11)
        with self.assertRaises(KeyError):
            b = a[11]
        a[11] = 11
        a.delete(10)
        with self.assertRaises(KeyError):
            b = a[10]
        a[1] = 1
        a[2] = 2
        a[3] = 3
        a[4] = 4
        a[12] = 12
        a.delete(2)
        with self.assertRaises(KeyError):
            b = a[2]
        self.assertEqual(a.root.left.value, 3)
        self.assertEqual(a.root.left.left.value, 1)
        self.assertEqual(a.root.left.right.value, 4)
    
    def test_traverse(self):
        a = avl.AVL_Tree()
        a[5] = 5
        a[7] = 7
        a[9] = 9
        a[8] = 8
        a[10] = 10
        a[11] = 11
        results = []
        for key in a:
            results.append(key)
        self.assertEqual(results, [5, 7, 8, 9, 10, 11])
        results2 = []
        for key, value in a.items():
            results2.append((key, value))
        self.assertEqual(results2, [(5, 5), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11)])


if __name__ == '__main__':
    unittest.main()
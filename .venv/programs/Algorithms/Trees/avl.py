class Node():
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.parent = None
        self.left = None
        self.right = None
        self.height = 1


class AVL_Iterator():
    def __init__(self, root):
        self.current = root

    def __iter__(self):
        return self

    def __next__(self):
        if self.current:
            item = self.current.value
            self.current = self.current.next
            return item
        else:
            raise StopIteration


class AVL_Tree():
    def __init__(self, *items):
        self.root = None
        for key, value in items:
            self.insert(key, value)


    def _right_rotation(self, old_root):
        new_root = old_root.left
        old_root.left = new_root.right
        if new_root.right:
            new_root.right.parent = old_root
        
        if old_root.parent:
            if old_root.parent.left == old_root:
                old_root.parent.left = new_root
            else:
                old_root.parent.right = new_root
        else:
            self.root = new_root

        new_root.right = old_root
        new_root.parent = old_root.parent
        old_root.parent = new_root
        old_root.height = 1 + max(self._get_height(old_root.right), self._get_height(old_root.left))
        new_root.height = 1 + max(self._get_height(new_root.right), self._get_height(new_root.left))

        return new_root
    
    
    def _left_rotation(self, old_root):
        new_root = old_root.right
        old_root.right = new_root.left
        if new_root.left:
            new_root.left.parent = old_root
        
        if old_root.parent:
            if old_root.parent.left == old_root:
                old_root.parent.left = new_root
            else:
                old_root.parent.right = new_root
        else:
            self.root = new_root

        new_root.left = old_root
        new_root.parent = old_root.parent
        old_root.parent = new_root
        old_root.height = 1 + max(self._get_height(old_root.right), self._get_height(old_root.left))
        new_root.height = 1 + max(self._get_height(new_root.right), self._get_height(new_root.left))

        return new_root


    def _get_height(self, root):
        if not root:
            return 0
        
        return root.height

    
    def _get_balance_factor(self, root):
        if not root:
            return 0
        
        return self._get_height(root.left) - self._get_height(root.right)
    
    
    def _insert(self, root, key, value):
        if not root:
            return Node(key, value)

        if key < root.key:
            left_sub_root = self._insert(root.left, key, value)
            root.left = left_sub_root
            left_sub_root.parent = root
        elif key == root.key:
            root.value = value
            return root
        else:
            right_sub_root = self._insert(root.right, key, value)
            root.right = right_sub_root
            right_sub_root.parent = root
        
        root.height = 1 + max(self._get_height(root.left), self._get_height(root.right))
        
        return self._rebalance(root)


    def _rebalance(self, root):
        balance_factor = self._get_balance_factor(root)

        if balance_factor > 1:
            if self._get_balance_factor(root.left) < 0:
                root.left = self._left_rotation(root.left)
            return self._right_rotation(root)

        elif balance_factor < -1:
            if self._get_balance_factor(root.right) > 0:
                root.right = self._right_rotation(root.right)
            return self._left_rotation(root)

        else:
            return root


    def _transplant(self, deleting_node, replacing_node=None):
        if not deleting_node.parent:
            self.root = replacing_node

        elif deleting_node == deleting_node.parent.left:
            deleting_node.parent.left = replacing_node
        else:
            deleting_node.parent.right = replacing_node

        if replacing_node:
            replacing_node.parent = deleting_node.parent




    def __getitem__(self, key):
        node = self._search(key)
        if node:
            return node.value
        else:
            raise KeyError('Key not found in tree')       


    def __setitem__(self, key, value):
        if not self.root:
            new_node = Node(key, value)
            self.root = new_node
            return

        self._insert(self.root, key, value)           


    def delete(self, key):
        return self._delete_helper(key, self.root)


    def _delete_helper(self, key, root):
        if not root:
            return root

        elif key < root.key:
            root.left = self._delete_helper(key, root.left)

        elif key > root.key:
            root.right = self._delete_helper(key, root.right)

        else:
            if not root.left:
                tmp = root.right
                root = None
                return tmp

            elif not root.right:
                tmp = root.left
                root = None
                return tmp

            tmp = self.successor(root.right)
            root.key = tmp.key
            root.right = self._delete_helper(tmp.key, root.right)

        if not root:
            return root

        root.height = 1 + max(self._get_height(root.left), self._get_height(root.right))
        
        return self._rebalance(root)



    def _search(self, key):
        cur_node = self.root
        while cur_node and key != cur_node.key:
            if key < cur_node.key:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right

        return cur_node


    def min(self, root):
        cur_node = root

        while cur_node and cur_node.left:
            cur_node = cur_node.left
        
        return cur_node


    def max(self, root):
        cur_node = root
        if cur_node:
            while cur_node.right:
                cur_node = cur_node.right
        
        return cur_node


    def predecessor(self, node):
        pass


    def successor(self, key):
        return self._successor_helper(self._search(key))


    def _successor_helper(self, node):
        if node.right:
            return self.min(node.right)

        parent_node = node.parent
        cur_node = node
        while parent_node:
            if cur_node is parent_node.left:
                break
            cur_node = parent_node
            parent_node = parent_node.parent
        return parent_node


    def __iter__(self):
        return AVL_Iterator(self.root)

    
    def __contains__(self, item):
        if self.__search__(item):
            return True
        else:
            return False


    def os_select(self, root, i):
        rank = root.left.size + 1
        if i == rank:
            return root
        elif i < rank:
            return self.os_select(root.left, i)
        else:
            return self.os_select(root.right, i)

    
    def os_rank(self, key):
        queried_node = self.search(key)
        rank = queried_node.left.size + 1
        
        tmp_node = queried_node
        while tmp_node != self.root:
            if tmp_node.key == tmp_node.parent.right.key:
                rank += tmp_node.parent.left.size + 1
            tmp_node = tmp_node.parent
        
        return rank
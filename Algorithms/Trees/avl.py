class Node():
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.parent = None
        self.left = None
        self.right = None
        self.height = 1


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
        if self.root and (deleting_node := self._search(key)):
            # Case: no child
            if (deleting_node.left is None) and (deleting_node.right is None):
                self._delete_no_child(deleting_node=deleting_node)
            # Case: Two children
            elif deleting_node.left and deleting_node.right:
                replacing_node = self._successor_helper(deleting_node)
                # Replace the deleting node with the replacing node, but keep the replacing node in place.
                deleting_node.key = replacing_node.key
                deleting_node.value = replacing_node.value
                if replacing_node.right:  # The replacing node cannot have left child.
                    self._delete_one_child(deleting_node=replacing_node)
                else:
                    self._delete_no_child(deleting_node=replacing_node)
            # Case: one child
            else:
                self._delete_one_child(deleting_node=deleting_node)


    def _delete_no_child(self, deleting_node):
        parent = deleting_node.parent
        self._transplant(deleting_node)
        if parent:
            self._delete_fixup(parent)


    def _delete_one_child(self, deleting_node):
        parent = deleting_node.parent
        replacing_node = (deleting_node.right if deleting_node.right else deleting_node.left)
        self._transplant(deleting_node, replacing_node)
        if parent:
            self._delete_fixup(parent)

    
    def _delete_fixup(self, fixing_node):
        while fixing_node:
            fixing_node.height = 1 + max(self._get_height(fixing_node.left), self._get_height(fixing_node.right))

            if self._get_balance_factor(fixing_node) > 1:
                # Case Left-Left
                if self._get_balance_factor(fixing_node.left) >= 0:
                    self._right_rotation(fixing_node)
                # Case Left-Right
                elif self._get_balance_factor(fixing_node.left) < 0:
                    # The fixing node's left child cannot be empty
                    self._left_rotation(fixing_node.left)
                    self._right_rotation(fixing_node)
            elif self._get_balance_factor(fixing_node) < -1:
                # Case Right-Right
                if self._get_balance_factor(fixing_node.right) <= 0:
                    self._left_rotation(fixing_node)
                # Case Right-Left
                elif self._get_balance_factor(fixing_node.right) > 0:
                    # The fixing node's right child cannot be empty
                    self._right_rotation(fixing_node.right)
                    self._left_rotation(fixing_node)

            fixing_node = fixing_node.parent


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
        return self._inorder_traverse(self.root)

    
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

    
    def _inorder_traverse(self, node):
        if node:
            yield from self._inorder_traverse(node.left)
            yield node.key
            yield from self._inorder_traverse(node.right)
    
    
    def items(self):
        return self._inorder_traverse_items(self.root)


    def _inorder_traverse_items(self, node):
        if node:
            yield from self._inorder_traverse_items(node.left)
            yield (node.key, node.value)
            yield from self._inorder_traverse_items(node.right)

    
    def __str__(self):
        contents = [item for item in self._inorder_traverse(self.root)]
        return str(contents)
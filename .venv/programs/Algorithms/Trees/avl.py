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


    def __right_rotation__(self, old_root):
        new_root = old_root.left
        if new_root.right:
            old_root.left = new_root.right
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
        old_root.height = 1 + max(self.__get_height__(old_root.right), self.__get_height__(old_root.left))
        new_root.height = 1 + max(self.__get_height__(new_root.right), self.__get_height__(new_root.left))
    
    
    def __left_rotation__(self, old_root):
        new_root = old_root.right
        if new_root.left:
            old_root.right = new_root.left
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
        old_root.height = 1 + max(self.__get_height__(old_root.right), self.__get_height__(old_root.left))
        new_root.height = 1 + max(self.__get_height__(new_root.right), self.__get_height__(new_root.left))
    
    
    def __right_left_rotation__(self, old_root):
        self.__right_rotation__(old_root.right)
        self.__left_rotation__(old_root)
    
    
    def __left_right_rotation__(self, old_root):
        self.__left_rotation__(old_root.left)
        self.__right_rotation__(old_root)


    def __get_height__(self, root):
        if not root:
            return 0
        
        return root.height

    
    def __get_balance_factor__(self, root):
        if not root:
            return 0
        
        return self.__get_height__(root.left) - self.__get_height__(root.right)
    
    
    def insert(self, key, value):
        new_node = Node(key, value)
        
        if not self.root:
            self.root = new_node
            return
        
        cur_node = self.root
        prev_node = None
        while cur_node:
            prev_node = cur_node
            if new_node.key == cur_node.key:
                cur_node.value = new_node.value
                return
            elif new_node.key < cur_node.key:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right

        if new_node.key < cur_node.key:
            prev_node.left = new_node
        else:
            prev_node.right = new_node
        
        new_node.parent = prev_node

        balance = self.__get_balance_factor__(new_node)







    def __transplant__(self, deleting_node, replacing_node=None):
        if not deleting_node.parent:
            self.root = replacing_node

        elif deleting_node == deleting_node.parent.left:
            deleting_node.parent.left = replacing_node
        else:
            deleting_node.parent.right = replacing_node

        if replacing_node:
            replacing_node.parent = deleting_node.parent




    def __getitem__(self, key):
        return self.search(key).value
        #return getattr(self, key)        


    def __setitem__(self, key, value):
        setattr(self, key, value)            









    def delete(self, key):
        pass


    def search(self, key):
        cur_node = self.root
        while cur_node and key != cur_node.key:
            if key < cur_node.key:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right

        return cur_node


    def min(self):
        cur_node = self.root
        if cur_node:
            while cur_node.left:
                cur_node = cur_node.left
        
        return cur_node


    def max(self):
        cur_node = self.root
        if cur_node:
            while cur_node.right:
                cur_node = cur_node.right
        
        return cur_node


    def predecessor(self, value):
        pass


    def successor(self, value):
        pass


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
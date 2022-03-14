class Node():
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.parent = None
        self.left = None
        self.right = None
        self.balance_factor = 0
        self.height = 0


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


class AVL():
    def __init__(self, *items):
        self.root = None


    def insert(self, key, value):
        new_node = Node(key, value)
        if not self.root:
            self.root = new_node
        else:
            cur_node = self.root
            prev_node = None
            while cur_node:
                prev_node = cur_node
                if new_node.key < cur_node.key:
                    cur_node = cur_node.left
                else:
                    cur_node = cur_node.right

            


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
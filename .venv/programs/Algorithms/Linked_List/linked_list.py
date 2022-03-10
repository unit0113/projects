class Node():
    def __init__(self, item):
        self.value = item
        self.next = None
        self.prev = None


class Linked_List():
    def __init__(self, *args):
        self.length = 0
        self.head = None
        self.tail = None
        for arg in args:
            self.add_front(arg)

    
    def add_front(self, *items):
        if not items:
            raise ValueError('add_front requires at least one argument')

        for item in items:
            new_node = Node(item)
            new_node.next = self.head
            
            if self.head:
                self.head.prev = new_node

            if not self.tail:
                self.tail = new_node

            self.head = new_node
            self.length += 1


    def add_rear(self, *items):
        if not items:
            raise ValueError('add_rear requires at least one argument')

        for item in items:
            new_node = Node(item)
            
            if self.tail:
                self.tail.next = new_node
                new_node.prev = self.tail

            if not self.head:
                self.head = new_node

            self.tail = new_node
            self.length += 1


    def remove_first(self):
        if self.length == 0:
            raise ValueError('Linked List is empty')

        tmp = self.head
        if self.length == 1:
            self.head = None
            self.tail = None
        else:
            self.head.next.prev = None
            self.head = self.head.next

        self.length -= 1
        return tmp.value
        

    def remove_last(self):
        if self.length == 0:
            raise ValueError('Linked List is empty')

        tmp = self.tail
        if self.length == 1:
            self.head = None
            self.tail = None
        else:
            self.tail.prev.next = None
            self.tail = self.tail.prev

        self.length -= 1
        return tmp.value        
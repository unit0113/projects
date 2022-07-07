class Node():
    def __init__(self, item):
        self.value = item
        self.next = None
        self.prev = None


class Linked_List_Iterator():
    def __init__(self, head):
        self.current = head

    def __iter__(self):
        return self

    def __next__(self):
        if self.current:
            item = self.current.value
            self.current = self.current.next
            return item
        else:
            raise StopIteration


class Linked_List():
    def __init__(self, *args):
        self.length = 0
        self.sentinel = Node(None)
        for arg in args:
            self.add_front(arg)

    
    def add_front(self, *items):
        if not items:
            raise ValueError('add_front requires at least one argument')

        for item in items:
            new_node = Node(item)
            new_node.next = self.sentinel.next
            new_node.prev = self.sentinel
            
            if self.length > 0:
                self.sentinel.next.prev = new_node
            else:
                self.sentinel.prev = new_node

            self.sentinel.next = new_node
            self.length += 1


    def add_rear(self, *items):
        if not items:
            raise ValueError('add_rear requires at least one argument')

        for item in items:
            new_node = Node(item)
            new_node.next = self.sentinel
            new_node.prev = self.sentinel.prev

            if self.length > 0:
                self.sentinel.prev.next = new_node
            else:
                self.sentinel.next = new_node

            self.sentinel.prev = new_node
            self.length += 1


    def remove_first(self):
        if self.length == 0:
            raise ValueError('Linked List is empty')

        tmp = self.sentinel.next
        if self.length == 1:
            self.sentinel.next = None
            self.sentinel.prev = None
        else:
            self.sentinel.next.next.prev = self.sentinel
            self.sentinel.next = self.sentinel.next.next

        self.length -= 1
        return tmp.value
        

    def remove_last(self):
        if self.length == 0:
            raise ValueError('Linked List is empty')

        tmp = self.sentinel.prev
        if self.length == 1:
            self.sentinel.next = None
            self.sentinel.prev = None
        else:
            self.sentinel.prev.prev.next = self.sentinel
            self.sentinel.prev = self.sentinel.prev.prev

        self.length -= 1
        return tmp.value


    def __search__(self, item):
        cur_node = self.sentinel.next
        while cur_node:
            if item == cur_node.value:
                return cur_node
            cur_node = cur_node.next

        return False


    def find(self, item):
        return self.__search__(item) != False

    
    def remove(self, item):
        node_to_remove = self.__search__(item)
        if node_to_remove:
            if self.sentinel.next == node_to_remove:
                self.remove_first()
            elif self.sentinel.prev == node_to_remove:
                self.remove_last()
            else:
                prev = node_to_remove.prev
                next = node_to_remove.next
                prev.next = next
                next.prev = prev

    
    def peek(self):
        return self.sentinel.next.value

    
    def peek_tail(self):
        return self.sentinel.prev.value


    def __iter__(self):
        return Linked_List_Iterator(self.sentinel.next)

    
    def __contains__(self, item):
        if self.__search__(item):
            return True
        else:
            return False
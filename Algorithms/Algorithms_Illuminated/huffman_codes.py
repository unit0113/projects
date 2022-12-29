import heapq
import itertools


class Node:
    def __init__(self, item: object, weight: int) -> None:
        self.item = item
        self.weight = weight
        self.left = None
        self.right = None

    def set_children(self, ln, rn) -> None:
        self.left = ln
        self.right = rn

    def __repr__(self) -> str:
        if self.item == None:
            return f"Intermediate Node, combined weight: {self.weight}"
        return f"Item: {self.item}, Weight: {self.weight}, Left: {self.left}, Right: {self.right}"

    def __lt__(self, other: object) -> bool:
        return self.weight < other.weight

    def __eq__(self, other: object) -> bool:
        return self.weight == other.weight



class Huffman:
    def __init__(self) -> None:
        self.root = None
        self.tasks = []


    def add(self, node: Node) -> None:
        heapq.heappush(self.tasks, node)

    def greedy_huffman(self) -> None:
        tasks_copy = self.tasks.copy()
        while len(tasks_copy) > 1:
            left = heapq.heappop(tasks_copy)
            right = heapq.heappop(tasks_copy)
            new_freq = left.weight + right.weight
            new_node = Node(None, new_freq)
            new_node.set_children(left, right)
            heapq.heappush(tasks_copy, new_node)

        self.root = heapq.heappop(tasks_copy)

    def _in_order(self, node) -> str:
        if node.left:
            self._in_order(node.left)
        print(node)
        if node.right:
            self._in_order(node.right)

    def print(self) -> None:
        if self.tasks:
            self.greedy_huffman()
            self._in_order(self.root)
        else:
            print("Tree is empty")


if __name__ == '__main__':
    a = Node('a', 42)
    b = Node('b', 20)
    c = Node('c', 5)
    d = Node('d', 10)
    e = Node('e', 11)
    f = Node('f', 12)

    huff = Huffman()
    huff.print()
    huff.add(a)
    huff.add(b)
    huff.add(c)
    huff.add(d)
    huff.add(e)
    huff.add(f)

    huff.print()

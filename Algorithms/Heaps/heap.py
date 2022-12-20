from typing import Optional


class Heap:
    def __init__(self) -> None:
        self.heap = []

    def push(self, item) -> None:
        self.heap.append(item)
        curr_index = len(self.heap) - 1
        parent_index = self._get_parent_index(curr_index)

        while parent_index != None and self.heap[curr_index] < self.heap[parent_index]:
            self._swap(parent_index, curr_index)
            curr_index = parent_index
            parent_index = self._get_parent_index(curr_index)

    def peek(self):
        if self.is_empty():
            return None

        return self.heap[0]

    def pop(self):
        if self.is_empty():
            return None

        result = self.heap[0]
        self._remove(0)
        return result

    def heapify(self, data: list) -> None:
        for item in data:
            self.push(item)

    def remove(self, data) -> None:
        if data in self.heap:
            index = self.heap.index(data)
            self._remove(index)

    def _remove(self, index: int) -> int:
        self.heap[index] = self.heap[-1]
        del self.heap[-1]

        curr_index = index
        min_child_index = self._find_min_child_index(curr_index)
        while min_child_index and self.heap[curr_index] > self.heap[min_child_index]:
            self._swap(curr_index, min_child_index)
            curr_index = min_child_index
            min_child_index = self._find_min_child_index(curr_index)        
        
    def _find_min_child_index(self, index: int) -> Optional[int]:
        l_child_index = self._get_left_child(index)
        r_child_index = self._get_right_child(index)

        if l_child_index and r_child_index:
            return l_child_index if self.heap[l_child_index] <= self.heap[r_child_index] else r_child_index
        elif l_child_index:
            return l_child_index
        elif r_child_index:
            return r_child_index
        else:
            return None

    def print(self) -> None:
        print(self.heap)

    def is_empty(self) -> bool:
        return not self.heap

    def _get_parent_index(self, index: int) -> Optional[int]:
        return (index - 1) // 2 if index != 0 else None

    def _get_left_child(self, index: int) -> Optional[int]:
        child = 2 * index + 1
        return child if child < len(self.heap) else None

    def _get_right_child(self, index: int) -> Optional[int]:
        child = 2 * index + 2
        return child if child < len(self.heap) else None

    def _swap(self, pos1: int, pos2: int) -> None:
        self.heap[pos1], self.heap[pos2] = self.heap[pos2], self.heap[pos1]

    def __len__(self) -> int:
        return len(self.heap)

    def __bool__(self) -> bool:
        return self.heap


if __name__ == "__main__":
    heap = Heap()
    heap.push(5)
    heap.push(2)
    heap.push(1)
    heap.push(4)
    heap.push(9)
    heap.push(3)
    heap.push(12)
    heap.print()
    heap.pop()
    heap.print()
    heap.heapify([4, 1, 7, 76])
    heap.print()
    
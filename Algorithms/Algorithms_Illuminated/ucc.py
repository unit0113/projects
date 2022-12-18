import queue


def ucc(graph: dict[int, list[int]]) -> int:
    ucc_count = 0
    explored = []
    q = queue.Queue()

    for vertex in graph:
        if vertex not in explored:
            ucc_count += 1
            q.queue.clear()
            q.put(vertex)

            while not q.empty():
                start = q.get()
                for v in graph[start]:
                    if v not in explored:
                        q.put(v)
                        explored.append(v)

    return ucc_count


graph = {1: [3, 5], 3: [1, 5], 5: [1, 3, 7, 9], 7: [5], 9: [5], 2: [4], 4: [2], 8: [6], 6: [8, 10], 10: [6]}
print(ucc(graph))
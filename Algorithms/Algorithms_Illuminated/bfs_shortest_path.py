from queue import Queue


class Vertex:
    def __init__(self, name: str, connections: list[str]) -> None:
        self.name = name
        self.path = []
        self.connections = connections
        self.explored = False

    def __repr__(self) -> str:
        return self.name


def shortest_path(graph: dict, start: str, end: str) -> bool:
    if start == end: return 0
    
    explored = [start]
    q = Queue()
    q.put(start)

    while not q.empty():
        exploring = q.get()
        for v in graph[exploring].connections:
            if v == end:
                graph[v].path = graph[exploring].path + [exploring] + [v]
                return graph[v].path
            if v not in explored:
                graph[v].path = graph[exploring].path + [exploring]
                q.put(v)
                explored.append(v)

    return False


adj_list = {'s': Vertex('s', ['a', 'b']), 'a': Vertex('a', ['s', 'c']), 'b': Vertex('b', ['s', 'c', 'd']), 'c': Vertex('c', ['a', 'b', 'd', 'e']), 'd': Vertex('d', ['b', 'c', 'e']), 'e': Vertex('e', ['c', 'd'])}
print(shortest_path(adj_list, 's', 'e'))
print(shortest_path(adj_list, 's', 'f'))
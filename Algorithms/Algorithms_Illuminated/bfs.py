from queue import Queue

def bfs(graph: dict, source: str, target: str) -> bool:
    if source == target: return True
    
    explored = [source]
    q = Queue()
    q.put(source)

    while not q.empty():
        start = q.get()
        for v in graph[start]:
            if v == target:
                return True
            if v not in explored:
                q.put(v)
                explored.append(v)

    return False


adj_list = {'s': ['a', 'b'], 'a': ['s', 'c'], 'b': ['s', 'c', 'd'], 'c': ['a', 'b', 'd', 'e'], 'd': ['b', 'c', 'e'], 'e': ['c', 'd']}
print(bfs(adj_list, 's', 'e'))
print(bfs(adj_list, 's', 'f'))
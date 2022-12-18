from collections import deque

def dfs(graph: dict, source: str, target: str) -> bool:
    if source == target: return True
    
    explored = [source]
    s = deque()
    s.append(source)

    while s:
        start = s.pop()
        for v in graph[start]:
            if v == target:
                return True
            if v not in explored:
                s.append(v)
                explored.append(v)

    return False


adj_list = {'s': ['a', 'b'], 'a': ['s', 'c'], 'b': ['s', 'c', 'd'], 'c': ['a', 'b', 'd', 'e'], 'd': ['b', 'c', 'e'], 'e': ['c', 'd']}
print(dfs(adj_list, 's', 'e'))
print(dfs(adj_list, 's', 'f'))
from collections import deque

def dfs(graph: dict, source: str, target: str, explored: list = []) -> bool:
    explored.append(source)
    
    for neighbor in graph[source]:
        if neighbor == target:
            return True
        elif neighbor not in explored:
            return dfs(graph, neighbor, target, explored)
    


adj_list = {'s': ['a', 'b'], 'a': ['s', 'c'], 'b': ['s', 'c', 'd'], 'c': ['a', 'b', 'd', 'e'], 'd': ['b', 'c', 'e'], 'e': ['c', 'd']}
print(dfs(adj_list, 's', 'e'))
print(dfs(adj_list, 's', 'f'))
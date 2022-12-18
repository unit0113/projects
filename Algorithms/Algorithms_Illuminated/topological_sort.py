def topological_sort(graph, node):
    result = []
    explored = []

    def recursive_helper(node):
        for neighbor in graph[node]:
            if neighbor not in explored:
                explored.append(neighbor)
                recursive_helper(neighbor)
        result.append(node)

    recursive_helper(node)
    return {vertex: index for index, vertex in enumerate(result[::-1])}


graph = {'s': ['v', 'w'], 'v': ['t'], 'w': ['t'], 't': []}
print(topological_sort(graph, 's'))
graph = {'a': ['b'], 'b': ['c'], 'c': ['d'], 'd': ['e'], 'e': ['f'], 'f': ['g'], 'g': []}
print(topological_sort(graph, 'a'))
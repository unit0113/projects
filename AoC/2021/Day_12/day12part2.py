adj_matrix = {}
with open(r'AoC\2021\Day_12\input.txt', 'r') as file:
    for line in file.readlines():
        src, dst = line.strip().split('-')
        if src not in adj_matrix:
            adj_matrix[src] = []
        if dst not in adj_matrix:
            adj_matrix[dst] = []
        adj_matrix[src].append(dst)
        adj_matrix[dst].append(src)

#adj_matrix = {'start': ['A', 'b'], 'A': ['start', 'c', 'b', 'end'], 'b': ['A', 'start', 'd', 'end'], 'c': ['A'], 'd': ['b'], 'end': ['A', 'b']}


def find_number_paths(graph):
    paths = set()
    starting_path = ('start',)


    def find_number_paths_helper(graph: dict, path: tuple):
        for dst in graph[path[-1]]:
            if can_add(path, dst):
                new_path = path + (dst,)
                if dst == 'end':
                    paths.add(new_path)
                    continue

                find_number_paths_helper(graph, new_path)


    find_number_paths_helper(graph, starting_path)

    return len(paths)


def can_add(path: tuple, dst:str):
    if dst == 'start':
        return False

    if dst not in path or dst.isupper() or dst == 'end':
        return True

    truncated_path = [dst for dst in path[1:] if dst.islower()]
    if len(set(truncated_path)) < len(truncated_path):
        return False
    
    return True


print(find_number_paths(adj_matrix))
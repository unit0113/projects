from collections import defaultdict


class Graph:
    def __init__(self) -> None:
        self.adj_list = {}

    @property
    def size(self):
        return len(self.adj_list)
    
    def __bool__(self):
        return self.adj_list
    
    @property
    def directed_edge_list(self):
        edges = []
        for source in self.adj_list.keys():
            for dest in self.adj_list[source]:
                edges.append((source, dest))
        
        return edges

    def add_vertex(self, edge_name):
        if edge_name not in self.adj_list.keys():
            self.adj_list[edge_name] = set()

    def add_undirected_edge(self, v_1, v_2):
        if v_1 not in self.adj_list.keys():
            raise ValueError('Vertex 1 not in graph')
        if v_2 not in self.adj_list.keys():
            raise ValueError('Vertex 2 not in graph')

        self.adj_list[v_1].add(v_2)
        self.adj_list[v_2].add(v_1)

    def add_directed_edge(self, source, dest):
        if source not in self.adj_list.keys():
            raise ValueError('Source vertex not in graph')
        if dest not in self.adj_list.keys():
            raise ValueError('Destination vertex not in graph')
        
        self.adj_list[source].add(dest)

    def _dfs_recursive(self, current, V, visited):
        V[current] = True

        for neighbor in self.adj_list[current]:
            if not V[neighbor]:
                self._dfs_recursive(neighbor, V, visited)
        visited.append(current)

    def topological_sort(self):
        V = {node: False for node in self.adj_list.keys()}
        ordering = []

        for current in self.adj_list.keys():
            if not V[current]:
                visited = []
                self._dfs_recursive(current, V, visited)
                for vertex in visited:
                    ordering.append(vertex)

        ordering.reverse()
        return ordering
    
    def topological_sort_kahn(self):
        incoming_edges = defaultdict(lambda: 0)
        for _, vertex in self.directed_edge_list:
            incoming_edges[vertex] += 1

        zero_queue = [vertex for vertex in self.adj_list.keys() if vertex not in incoming_edges.keys()]
        ordering = []
        while zero_queue:
            current = zero_queue.pop(0)
            ordering.append(current)
            for vertex in self.adj_list[current]:
                incoming_edges[vertex] -= 1
                if incoming_edges[vertex] == 0:
                    zero_queue.append(vertex)

        return ordering
    
    def _valid_topological_order(self, ordering):
        for source, dest in self.directed_edge_list:
            if ordering.index(source) > ordering.index(dest):
                return False
        return True

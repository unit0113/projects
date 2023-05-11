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

    def _dfs_recursive(self, current, visited):
        visited[current] = True

        for neighbor in self.adj_list[current]:
            if not visited[neighbor]:
                self._dfs_recursive(neighbor, visited)
        visited[current] = True

    def topological_sort(self):
        visited = {node: False for node in self.adj_list.keys()}
        ordering = []

        for current in self.adj_list.keys():
            if not visited[current]:
                self._dfs_recursive(current, visited)
                for vertex in visited:
                    ordering.append(vertex)

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

    def get_msccs(self):
        visited = []
        stack = []
        for vertex in self.adj_list.keys():
            if vertex not in visited:
                self._dfs_msccs(vertex, visited, stack)

        reversed_graph = self.get_reverse()
        visited = []
        msccs = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            seen = []
            reversed_graph._dfs_msccs(current, visited, seen)
            msccs.append(sorted(seen))
        
        return msccs

    def _dfs_msccs(self, vertex, visited, stack):
        visited.append(vertex)
        for u in self.adj_list[vertex]:
            if u not in visited:
                self._dfs_msccs(u, visited, stack)
        stack.append(vertex)

    def get_reverse(self):
        reversed_adj_list = defaultdict(set)
        for vertex in self.adj_list.keys():
            for edge in self.adj_list[vertex]:
                reversed_adj_list[edge].add(vertex)
        
        reversed_graph = self
        reversed_graph.adj_list = reversed_adj_list
        return reversed_graph

    def get_min_spanning_tree(self):
        pass
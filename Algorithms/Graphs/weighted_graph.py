from typing import Self
from heapq import heapify, heappop


class UnionFind:
    def __init__(self, vertices=None) -> None:
        self.parents = {}
        self.ranks = {}

        if vertices:
            self.unify(vertices)

    def unify(self, vertices: list) -> None:
        for vertex in vertices:
            self.add(vertex)

    def add(self, vertex) -> None:
        self.parents[vertex] = vertex
        self.ranks[vertex] = 0

    def find(self, vertex) -> object:
        if self.parents[vertex] != vertex:
            self.parents[vertex] = self.find(self.parents[vertex])

        return self.parents[vertex]
    
    def union(self, v_1, v_2) -> None:
        if self.ranks[v_1] < self.ranks[v_2]:
            self.parents[self.find(v_1)] = self.find(v_2)
        elif self.ranks[v_2] < self.ranks[v_1]:
            self.parents[self.find(v_2)] = self.find(v_1)
        else:
            self.parents[self.find(v_2)] = v_1
            self.ranks[v_1] += 1

class Edge:
    def __init__(self, source, dest, weight) -> None:
        self._source = source
        self._dest = dest
        self._weight = weight

    @property
    def source(self):
        return self._source
    
    @property
    def dest(self):
        return self._dest
    
    @property
    def weight(self):
        return self._weight
    
    def __lt__(self, other: Self) -> bool:
        return self.weight < other.weight

    def __eq__(self, other: Self) -> bool:
        return self.weight == other.weight
    
    def __str__(self) -> str:
        return f'\tSource Vertex: {self.source}\n\tDestination Vertex: {self.dest}\n\tPath Weight: {self.weight}'


class WeightedGraph:
    def __init__(self) -> None:
        self.adj_list = {}

    @property
    def size(self) -> int:
        return len(self.adj_list)
    
    def __bool__(self):
        return self.adj_list
    
    @property
    def directed_edge_list(self) -> list[Edge]:
        edges = []
        for source in self.adj_list.keys():
            for dest, weight in self.adj_list[source]:
                edges.append(Edge(source, dest, weight))
        
        return edges

    def add_vertex(self, edge_name) -> None:
        if edge_name not in self.adj_list.keys():
            self.adj_list[edge_name] = set()

    def add_undirected_edge(self, v_1, v_2, weight) -> None:
        if v_1 not in self.adj_list.keys():
            raise ValueError('Vertex 1 not in graph')
        if v_2 not in self.adj_list.keys():
            raise ValueError('Vertex 2 not in graph')

        self.adj_list[v_1].add((v_2, weight))
        self.adj_list[v_2].add((v_1, weight))

    def add_directed_edge(self, source, dest, weight) -> None:
        if source not in self.adj_list.keys():
            raise ValueError('Source vertex not in graph')
        if dest not in self.adj_list.keys():
            raise ValueError('Destination vertex not in graph')
        
        self.adj_list[source].add((dest, weight))

    def get_min_spanning_tree(self) -> list[Edge]:
        minheap = self.directed_edge_list
        heapify(minheap)
        uf = UnionFind(list(self.adj_list.keys()))

        spanning_tree = []
        cost = 0
        num_nodes = len(self.adj_list.keys())
        while len(spanning_tree) < num_nodes - 1:
            curr_edge = heappop(minheap)
            if uf.find(curr_edge.source) != uf.find(curr_edge.dest):
                spanning_tree.append(curr_edge)
                cost += curr_edge.weight
                uf.union(curr_edge.source, curr_edge.dest)

        return spanning_tree, cost
    
    def print_min_spanning_tree(self) -> None:
        spanning_tree, cost = self.get_min_spanning_tree()
        print('Spanning Tree Details:')
        print(f'Cost: {cost}')
        for index, edge in enumerate(spanning_tree):
            print(f'\nEdge {index + 1}:')
            print(edge)

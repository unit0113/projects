import heapq
import math

def prim(adj_list: dict[list[tuple[str, int]]]) -> list:
    nodes = (list(adj_list.keys()))
    mst = [nodes[0]]
    U = set(nodes[0])
    
    for key in adj_list:
        heapq.heapify(adj_list[key])
        
    while len(U) < len(nodes):
        min_weight = math.inf
        min_node = None
        min_edge = None
        for node in U:
            if not adj_list[node]:
                continue

            poss_min_edge = min(adj_list[node], key=lambda t: t[0])
            if poss_min_edge[0] < min_weight:
                min_weight = poss_min_edge[0]
                min_node = node
                min_edge = poss_min_edge

        adj_list[min_node].remove(min_edge)

        if min_edge[1] in U:
            continue
        if min_edge[1] != min_node:
            adj_list[min_edge[1]].remove((min_edge[0], min_node))

        mst.append(min_edge)
        U.add(min_edge[1])
        #adj_list[node].remove(min_edge)

    return mst


def mst_weight(mst: list) -> int:
    total = 0
    for weight, _ in mst[1:]:
        total += weight
    return total


adj_list = {'a': [(6, 'b'), (9, 'b'), (3, 'e'), (7, 'f')],
            'b': [(6, 'a'), (9, 'a'), (5, 'c'), (2, 'd'), (4, 'e')],
            'c': [(5, 'b'), (2, 'd')],
            'd': [(2, 'b'), (2, 'c'), (3, 'e')],
            'e': [(3, 'a'), (4, 'b'), (3, 'd'), (1, 'e'), (8, 'f')],
            'f': [(7, 'a'), (8, 'e')]
            }

mst = prim(adj_list)
print(mst)
print(f"Total MST weight: {mst_weight(mst)}")
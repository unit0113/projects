import heapq


def dijkstra(graph, starting_vertex):
    # Initialize all distances to inf, set starting distance to 0
    distances = {vertex: float('infinity') for vertex in graph}
    distances[starting_vertex] = 0

    heap = [(0, starting_vertex)]
    while heap:
        # Get edge with smallest weight
        current_distance, current_vertex = heapq.heappop(heap)

        # Ignore if worse than current calculated distance to node
        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            # Add to heap if better than current distance to that node
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))

    return distances


example_graph = {
    'U': {'V': 2, 'W': 5, 'X': 1},
    'V': {'U': 2, 'X': 2, 'W': 3},
    'W': {'V': 3, 'U': 5, 'X': 3, 'Y': 1, 'Z': 5},
    'X': {'U': 1, 'V': 2, 'W': 3, 'Y': 1},
    'Y': {'X': 1, 'W': 1, 'Z': 1},
    'Z': {'W': 5, 'Y': 1},
}
print(dijkstra(example_graph, 'X'))
# => {'U': 1, 'W': 2, 'V': 2, 'Y': 1, 'X': 0, 'Z': 2}
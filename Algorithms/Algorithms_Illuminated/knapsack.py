def knapsack(item_values: list[float], item_weights: list[float], capacity: int) -> list:
    n = len(item_values)
    memo = {}
    for c in range(capacity + 1):
        memo[(0, c)] = 0

    for i in range(1, n + 1):
        for c in range(capacity + 1):
            if item_weights[i-1] <= c:
                memo[(i, c)] = max(memo[(i-1, c)], item_values[i-1] + memo[(i-1, c - item_weights[i-1])])
            else:
                memo[(i, c)] = memo[(i-1, c)]

    return memo[(n, capacity)]


values = [500, 250, 1500, 1200, 1200, 800]
weights = [4, 3, 10, 12, 9, 6]
capacity = 30

print(knapsack(values, weights, capacity))

from Part1.program1 import program1
from Part2.program5B import program5B

from random import randint
import pandas as pd

W = 50
Ns = [
    250,
    500,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
    11000,
    12000,
    13000,
    14000,
    15000,
    16000,
    17000,
    18000,
    19000,
    20000,
]
REPEATS = 10

results = {"accuracy": []}

for n in Ns:
    # Initialize variables
    heights = set()
    while len(heights) < n:
        heights.add(randint(1, n * 5))

    heights = list(heights)

    widths = [randint(1, W) for _ in range(n)]

    # Run programs
    sum_hg = 0
    sum_ho = 0
    for _ in range(REPEATS):
        _, total_height_g, _ = program1(n, W, heights, widths)
        _, total_height_o, _ = program5B(n, W, heights, widths)
        sum_hg += total_height_g
        sum_ho += total_height_o
    results["accuracy"].append((sum_hg - sum_ho) / sum_ho)
    print(f"N={n}: {results['accuracy'][-1]}")


df = pd.DataFrame.from_dict(results, orient="index", columns=Ns)
df.to_csv("resultscomparision.csv")

from timeit import Timer
from random import randint
import pandas as pd

from program3 import program3
from program4 import program4
from program5A import program5A
from program5B import program5B


W = 10
Ns = [5, 10, 15, 20, 25, 30]
REPEATS = 10

results = {"program3": [], "program4": [], "program5A": [], "program5B": []}

for n in Ns:
    # Initialize variables
    heights = set()
    while len(heights) < n:
        heights.add(randint(1, n * 5))

    heights = list(heights)

    widths = [randint(1, W) for _ in range(n)]

    # Run programs
    results["program3"].append(
        Timer(lambda: program3(n, W, heights, widths)).timeit(REPEATS) * 1000 / REPEATS
    )
    print(f"Program 3, N={n}: {results['program3'][-1]} ms")

    results["program4"].append(
        Timer(lambda: program4(n, W, heights, widths)).timeit(REPEATS) * 1000 / REPEATS
    )
    print(f"Program 4, N={n}: {results['program4'][-1]} ms")

    results["program5A"].append(
        Timer(lambda: program5A(n, W, heights, widths)).timeit(REPEATS) * 1000 / REPEATS
    )
    print(f"Program 5A, N={n}: {results['program5A'][-1]} ms")

    results["program5B"].append(
        Timer(lambda: program5B(n, W, heights, widths)).timeit(REPEATS) * 1000 / REPEATS
    )
    print(f"Program 5B, N={n}: {results['program5B'][-1]} ms")


df = pd.DataFrame.from_dict(results, orient="index", columns=Ns)
df.to_csv("results4.csv")

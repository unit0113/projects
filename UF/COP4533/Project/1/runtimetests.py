from timeit import Timer
from random import randint, choice

from program1 import program1
from program2 import program2

W = 10
Ns = [250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
REPEATS = 10

for n in Ns:
    # Initialize variables
    heights = set()
    while len(heights) < n:
        heights.add(randint(1, n * 5))

    heights = sorted(list(heights))
    heights.reverse()
    widths = [randint(1, W) for _ in range(n)]

    # Run programs
    print(
        f"Program 1, N={n}: {Timer(lambda: program1(n, W, heights, widths)).timeit(REPEATS) * 1000 / REPEATS} ms"
    )
    # Rearrange heights to be unimodal
    minima = choice(range(int(n / 20), int(n * 19 / 20)))
    heights[minima:] = heights[n - 1 : minima : -1]
    print(
        f"Program 2, N={n}: {Timer(lambda: program2(n, W, heights, widths)).timeit(REPEATS) * 1000 / REPEATS} ms"
    )

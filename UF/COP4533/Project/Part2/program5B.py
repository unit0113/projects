from typing import List, Tuple


def program5B(
    n: int, W: int, heights: List[int], widths: List[int]
) -> Tuple[int, int, List[int]]:
    """
    Solution to Program 5B

    Parameters:
    n (int): number of sculptures
    W (int): width of the platform
    heights (List[int]): heights of the sculptures
    widths (List[int]): widths of the sculptures

    Returns:
    int: number of platforms used
    int: optimal total height
    List[int]: number of statues on each platform
    """

    # Initialize memo and set base case to 0
    memo = [float("inf")] * (n + 1)
    memo[0] = 0

    # Precalculate running sums for width
    width_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        width_sum[i] = width_sum[i - 1] + widths[i - 1]

    # DP loop
    for i in range(1, n + 1):
        max_height = 0
        for j in range(i, 0, -1):
            total_width = width_sum[i] - width_sum[j - 1]
            if total_width <= W:
                # For partition j to i, update the maximum height and memo
                max_height = max(max_height, heights[j - 1])
                memo[i] = min(memo[i], memo[j - 1] + max_height)
            else:
                break

    # Reconstruct solution
    platforms = []
    i = n
    while i > 0:
        for j in range(i, 0, -1):
            total_width = width_sum[i] - width_sum[j - 1]
            if total_width <= W and memo[i] == memo[j - 1] + max(heights[j - 1 : i]):
                platforms.append(i - j + 1)
                i = j - 1
                break

    platforms.reverse()

    return len(platforms), memo[n], platforms


if __name__ == "__main__":
    n, W = map(int, input().split())
    heights = list(map(int, input().split()))
    widths = list(map(int, input().split()))

    m, total_height, num_statues = program5B(n, W, heights, widths)

    print(m)
    print(total_height)
    for i in num_statues:
        print(i)

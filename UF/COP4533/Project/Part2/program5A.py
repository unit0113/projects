from typing import List, Tuple


def program5A(
    n: int, W: int, heights: List[int], widths: List[int]
) -> Tuple[int, int, List[int]]:
    """
    Solution to Program 5A

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
    memo = {}
    memo[0] = 0

    # Precalculate running sums for width
    width_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        width_sum[i] = width_sum[i - 1] + widths[i - 1]

    # Precompute max heights for each segment
    max_height_segment = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        max_height_segment[i][i] = heights[i]
        for j in range(i + 1, n):
            max_height_segment[i][j] = max(max_height_segment[i][j - 1], heights[j])

    # Recursive call
    def dp(i):
        # If already computed
        if i in memo:
            return memo[i]

        # Base case
        if i == 0:
            return 0

        min_height = float("inf")

        # Try all partitions from j to i
        for j in range(i, 0, -1):
            total_width = width_sum[i] - width_sum[j - 1]
            if total_width > W:
                break  # If the partition doesn't fit, stop exploring further

            # Get the maximum height in the current segment from j to i
            segment_max_height = max_height_segment[j - 1][i - 1]

            # Recursively calculate height for this partition and minimize total height
            height_for_this_partition = dp(j - 1) + segment_max_height
            min_height = min(min_height, height_for_this_partition)

        # Memoize the result for dp(i)
        memo[i] = min_height
        return min_height

    # Get recursion started
    total_height = dp(n)

    # Reconstruct solution
    platforms = []
    i = n
    while i > 0:
        for j in range(i, 0, -1):
            total_width = width_sum[i] - width_sum[j - 1]
            if total_width <= W and dp(i) == dp(j - 1) + max(heights[j - 1 : i]):
                platforms.append(i - j + 1)
                i = j - 1
                break

    platforms.reverse()

    return len(platforms), total_height, platforms


if __name__ == "__main__":
    n, W = map(int, input().split())
    heights = list(map(int, input().split()))
    widths = list(map(int, input().split()))

    m, total_height, num_statues = program5A(n, W, heights, widths)

    print(m)
    print(total_height)
    for i in num_statues:
        print(i)

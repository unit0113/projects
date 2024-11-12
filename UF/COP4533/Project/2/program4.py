from typing import List, Tuple


def program4(
    n: int, W: int, heights: List[int], widths: List[int]
) -> Tuple[int, int, List[int]]:
    """
    Solution to Program 4

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

    # Track the solution for reconstructing the platform allocation
    partition = [-1] * (n + 1)

    # DP loop
    for i in range(1, n + 1):
        for j in range(i, 0, -1):
            total_width = sum(widths[index] for index in range(j - 1, i))
            if total_width <= W:
                max_height_on_platform = max(heights[k] for k in range(j - 1, i))
                if memo[j - 1] + max_height_on_platform < memo[i]:
                    memo[i] = memo[j - 1] + max_height_on_platform
                    partition[i] = j - 1

    # Reconstruct solution
    platforms = []
    sculpture_indices = n
    while sculpture_indices > 0:
        prev_sculpture_index = partition[sculpture_indices]
        platforms.append(sculpture_indices - prev_sculpture_index)
        sculpture_indices = prev_sculpture_index

    platforms.reverse()

    return len(platforms), memo[n], platforms


if __name__ == "__main__":
    n, W = map(int, input().split())
    heights = list(map(int, input().split()))
    widths = list(map(int, input().split()))

    m, total_height, num_statues = program4(n, W, heights, widths)

    print(m)
    print(total_height)
    for i in num_statues:
        print(i)

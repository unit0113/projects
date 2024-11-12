from typing import List, Tuple

# https://cs.stackexchange.com/questions/125069/proof-of-a-greedy-algorithm-used-for-a-variation-of-bin-packing-problem


def program1(
    n: int, W: int, heights: List[int], widths: List[int]
) -> Tuple[int, int, List[int]]:
    """
    Solution to Program 1

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

    # Initialize loop parameters
    curr_width = widths[0]
    total_height = heights[0]
    platforms = [1]

    # Loop through all scultpures
    for index in range(1, n):
        # If can fit on current platform
        if curr_width + widths[index] <= W:
            # Add to current platform
            curr_width += widths[index]
            platforms[-1] += 1

        # If can't fit on current platform
        else:
            # Initialize new platform
            curr_width = widths[index]
            total_height += heights[index]
            platforms.append(1)

    return len(platforms), total_height, platforms


if __name__ == "__main__":
    n, W = map(int, input().split())
    heights = list(map(int, input().split()))
    widths = list(map(int, input().split()))

    m, total_height, num_statues = program1(n, W, heights, widths)

    print(m)
    print(total_height)
    for i in num_statues:
        print(i)

# https://stackoverflow.com/questions/9088321/show-all-possible-groupings-of-a-list-given-only-the-amount-of-sublists-length

from typing import List, Tuple
from itertools import chain, combinations
from math import inf


def all_combinations(indexes):
    for n in range(1, len(indexes)):
        for splits in combinations(range(1, len(indexes)), n):
            result = []
            prev = None
            for split in chain(splits, [None]):
                result.append(indexes[prev:split])
                prev = split
            yield result


def is_valid_platform(widths, platforms, W):
    for platform in platforms:
        if sum([widths[i] for i in platform]) > W:
            return False
    return True


def calc_height(heights, platforms):
    height_platforms = [[heights[i1] for i1 in platform] for platform in platforms]
    return sum([max(platform) for platform in height_platforms])


def program3(
    n: int, W: int, heights: List[int], widths: List[int]
) -> Tuple[int, int, List[int]]:
    """
    Solution to Program 3

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

    best_height = inf
    best_platforms = None

    # Check every permutation of statues
    for platforms in all_combinations(list(range(n))):
        # Check if valid
        if is_valid_platform(widths, platforms, W):
            # Check if best
            height = calc_height(heights, platforms)
            if height < best_height:
                best_height = height
                best_platforms = platforms

    return (
        len(best_platforms),
        best_height,
        [len(platform) for platform in best_platforms],
    )


if __name__ == "__main__":
    n, W = map(int, input().split())
    heights = list(map(int, input().split()))
    widths = list(map(int, input().split()))

    m, total_height, num_statues = program3(n, W, heights, widths)

    print(m)
    print(total_height)
    for i in num_statues:
        print(i)

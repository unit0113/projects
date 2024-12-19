from typing import List, Tuple

# https://www.csc.liv.ac.uk/~epa/surveyhtml.html


def program2(
    n: int, W: int, heights: List[int], widths: List[int]
) -> Tuple[int, int, List[int]]:
    """
    Solution to Program 2

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
    platform_heights = [heights[0]]
    platforms = [1]
    unimodal = False

    # Loop through all scultpures
    for index in range(1, n):
        # If minima passed
        if heights[index] > heights[index - 1]:
            # Start part two
            unimodal = True
            break

        # If can fit on current platform
        if curr_width + widths[index] <= W:
            # Add to current platform
            curr_width += widths[index]
            platforms[-1] += 1

        # If can't fit on current platform
        else:
            # Initialize new platform
            curr_width = widths[index]
            platform_heights.append(heights[index])
            platforms.append(1)

    # Part two, start from end of input and go backwards
    if unimodal:
        # Initialize loop parameters
        reverse_curr_width = widths[-1]
        reverse_platform_heights = [heights[-1]]
        reverse_platforms = [1]

        # Loop through remaining unplaced scultpures, from the back
        for reverse_index in range(-2, index - n - 1, -1):
            # If can fit on current platform
            if reverse_curr_width + widths[reverse_index] <= W:
                # Add to current platform
                reverse_curr_width += widths[reverse_index]
                reverse_platforms[-1] += 1

            # If can't fit on current platform
            else:
                # Initialize new platform
                reverse_curr_width = widths[reverse_index]
                reverse_platform_heights.append(heights[reverse_index])
                reverse_platforms.append(1)

        # Reverse reverse_platforms (will now be in front to back order)
        reverse_platforms.reverse()

        # Check if last normal and first reverse platforms can be combined
        # AKA, the last platform on the forward part and the first platform
        # on the now reversed part can fit onto a single platform
        if curr_width + reverse_curr_width <= W:
            # Update platform count
            platforms[-1] += reverse_platforms[0]
            # Keep max height
            platform_heights[-1] = reverse_platform_heights[0]
            # Delete combined platform
            del reverse_platforms[0]
            del reverse_platform_heights[0]

        # Combine
        platforms.extend(reverse_platforms)
        platform_heights.extend(reverse_platform_heights)

    return len(platforms), sum(platform_heights), platforms


if __name__ == "__main__":
    n, W = map(int, input().split())
    heights = list(map(int, input().split()))
    widths = list(map(int, input().split()))

    m, total_height, num_statues = program2(n, W, heights, widths)

    print(m)
    print(total_height)
    for i in num_statues:
        print(i)

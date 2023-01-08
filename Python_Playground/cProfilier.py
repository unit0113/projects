import cProfile
import pstats

#tuna library to visualize externally
#scalene is an alternative

def fibonacci_d(num):
    solved_numbers = [0 for _ in range(num+1)]

    def fibonacci_helper(num):
        if num <= 1: return num
        if solved_numbers[num]: return solved_numbers[num]
        if solved_numbers[num - 1] == 0:
            solved_numbers[num - 1] = fibonacci_helper(num - 1)

        if solved_numbers[num - 2] == 0:
            solved_numbers[num - 2] = fibonacci_helper(num - 2)

        solved_numbers[num] = solved_numbers[num - 1] + solved_numbers[num - 2]

        return solved_numbers[num]

    return fibonacci_helper(num)


def fibonacci_i(n):
    first = 0
    second = 1
    if n == 1: return first
    if n == 2: return second

    for i in range(2, n):
        current = first + second
        first = second
        second = current

    return second

def fibonacci_r(num):
    if num <= 1: return num
    return fibonacci_r(num - 1) + fibonacci_r(num - 2)


if __name__ == "__main__":
    with cProfile.Profile() as profile:
        num = 30
        fibonacci_d(num)
        fibonacci_i(num)
        fibonacci_r(num)

    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()
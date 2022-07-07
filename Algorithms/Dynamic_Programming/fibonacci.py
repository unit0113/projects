def fibonacci(num):
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

print(fibonacci(20))
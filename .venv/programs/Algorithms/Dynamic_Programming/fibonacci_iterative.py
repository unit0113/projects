def fibonacci(n):
    first = 0
    second = 1
    if n == 1: return first
    if n == 2: return second

    for i in range(2, n):
        current = first + second
        first = second
        second = current

    return second

for i in range(1, 15):
    print(fibonacci(i))
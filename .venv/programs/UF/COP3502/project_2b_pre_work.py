def consecutive_fours(arr):
    if len(arr) < 4:
        return False

    initial_value = arr[0]
    count = 1
    for val in arr[1:]:
        if val == initial_value:
            count += 1
            if count >= 4:
                return True
        else:
            initial_value = val
            count = 1

    return False


def sum_by_parity(arr):
    sum_even = sum_odd = 0
    for index, val in enumerate(arr):
        if index % 2:
            sum_odd += val
        else:
            sum_even += val

    return [sum_even, sum_odd]


def expand_by_index(arr):
    results = []
    for index, val in enumerate(arr):
        results += [index] * val

    return results


def numerical_count(string):
    count = 0
    for char in list(string):
        if char.isdigit():
            count += 1

    return count

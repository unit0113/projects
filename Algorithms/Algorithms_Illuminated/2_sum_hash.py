def two_sum(arr: list[int], target: int) -> bool:
    items = {x: x for x in arr}
    for num in arr:
        if target - num in items:
            return True

    return False


arr = [34,12,76,15,54,85,24,2,64,68,35,32,84,47]
print(two_sum(arr, 27))
print(two_sum(arr, 28))
import math


def linear_search(arr, target):
    for item in arr:
        if item == target:
            return True
    
    return False


def jump_search(arr, target):
    jump_size = int(math.sqrt(len(arr)))
    prev = 0

    while prev+jump_size < len(arr) and arr[prev] < target:
        if arr[prev+jump_size] == target:
            return True

        prev += jump_size

    return linear_search(arr[prev-jump_size:prev], target)

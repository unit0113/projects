def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    return recursive_search(arr, target, low, high)


def recursive_search(arr, target, low, high):
    if low > high:
        return False

    mid = (high + low) // 2
    if arr[mid] == target:
        return True
    elif arr[mid] < target:
        return recursive_search(arr, target, mid+1, high)
    else:
        return recursive_search(arr, target, low, high-1)
def recursive_search(arr, target, low, high):
    if low >= high:
        return False

    mid = (high + low) // 2
    if arr[mid] == target:
        return True
    elif arr[mid] < target:
        return recursive_search(arr, target, mid+1, high)
    else:
        return recursive_search(arr, target, low, high-1)


def exponential_search(arr, target):
    if not arr:
        return False
    
    if arr[0] == target:
        return True

    index = 1
    while index < len(arr) and arr[index] <= target:
        index *= 2

    low = index // 2
    high = min(len(arr), index)

    return recursive_search(arr, target, low, high)
def interpolation_search(arr, target):
    if not arr:
        return False

    left, right = 0, len(arr) - 1
 
    while arr[right] != arr[left] and arr[left] <= target <= arr[right]: 
        mid = left + (target - arr[left]) * (right - left) // (arr[right] - arr[left])
 
        if target == arr[mid]:
            return True

        elif target < arr[mid]:
            right = mid - 1

        else:
            left = mid + 1
 
    return False
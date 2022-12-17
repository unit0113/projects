def count_inversions(arr: list, left: int = None, right: int = None) -> int:
    if left == None: left = 0
    if right == None: right = len(arr) - 1

    if left >= right: return 0

    mid = (left + right) // 2
    left_count = count_inversions(arr, left, mid)
    right_count = count_inversions(arr, mid+1, right)
    split_count = count_split_inversions(arr, left, mid, right)

    return left_count + right_count + split_count


def count_split_inversions(arr: list, left: int, mid: int, right: int) -> int:
    temp_arr = []
    i = left
    j = mid + 1
    inv_count = 0

    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            # Not an inversion pair
            temp_arr.append(arr[i])
            i += 1
        else:
            temp_arr.append(arr[j])
            inv_count += mid - i + 1
            j += 1
    
    # Add remaining elements to array
    temp_arr += arr[i:mid+1]
    temp_arr += arr[j:right+1]

    #transfering this back to the original array
    for i in range(left,right+1):
        arr[i] = temp_arr[i-left]

    return inv_count


array = [1, 3, 5, 2, 4, 6]
print(count_inversions(array))
array = [8, 5, 3, 1]
print(count_inversions(array))
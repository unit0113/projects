import random


def quick_select(arr: list, ith_order_stat: int, left: int = None, right: int = None) -> int:
    if not left: left = 0
    if not right: right = len(arr) - 1

    pivot_index = partition(arr, left, right)
    
    if pivot_index == ith_order_stat:
        return arr[pivot_index]
    elif ith_order_stat > pivot_index:
        return quick_select(arr, ith_order_stat, pivot_index + 1, right)
    else:
        return quick_select(arr, ith_order_stat, left, pivot_index - 1)


def partition(arr: list, left: int, right: int) -> int:
    pivot_index = random.randint(left, right)
    pivot = arr[pivot_index]
    arr[left], arr[pivot_index] = arr[pivot_index], arr[left]

    pivot_index = left
    for i in range(left + 1, right + 1):
        if arr[i] < pivot:
            pivot_index += 1
            arr[i], arr[pivot_index] = arr[pivot_index], arr[i]

    arr[pivot_index], arr[left] = arr[left], arr[pivot_index]
    
    return pivot_index


array = [5, 3, 7, 8, 12, 15, 2, 0, 16]
print(quick_select(array, 3))
print(quick_select(array, 6))
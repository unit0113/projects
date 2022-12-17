import random


def quick_sort(arr: list, left:int = None, right:int = None) -> None:
    if not left: left = 0
    if not right: right = len(arr) - 1

    if left >= right: return

    pivot_index = partition(arr, left, right)
    quick_sort(arr, left, pivot_index - 1)
    quick_sort(arr, pivot_index + 1, right)


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


array = random.sample(range(0, 101), 100)
print(array)
quick_sort(array)
print(array)
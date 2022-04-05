import random

def quick_sort(array):
    quick_sort_helper(array, 0, len(array) - 1)


def quick_sort_helper(array, first, last):
    if first < last:
        split_point = partition(array, first, last)

        quick_sort_helper(array, first, split_point - 1)
        quick_sort_helper(array, split_point +  1, last)


def partition(array, first, last):
    pivot_value = array[first]
    left_mark = first + 1
    right_mark = last

    done = False
    while not done:
        while left_mark <= right_mark and array[left_mark] <= pivot_value:
            left_mark += 1

        while array[right_mark] >= pivot_value and right_mark >= left_mark:
            right_mark -= 1

        if right_mark < left_mark:
            done = True

        else:
            tmp = array[left_mark]
            array[left_mark] = array[right_mark]
            array[right_mark] = tmp

    tmp = array[first]
    array[first] = array[right_mark]
    array[right_mark] = tmp

    return right_mark


array = random.sample(range(0, 51), 25)
quick_sort(array)
print(array)
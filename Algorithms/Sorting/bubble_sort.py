def bubble_sort(array):
    for i in range(len(array), 0, -1):
        swapped = False
        for j in range(1, i):
            if array[j - 1] > array[j]:
                swapped = True
                tmp = array[j]
                array[j] = array[j - 1]
                array[j - 1] = tmp
        if not swapped:
            break

    return array

array = [4,5,2,7,9,7,45,2,3,4,6,7,3,90]
print(bubble_sort(array))
def insertion_sort(array):
    for i in range(1, len(array)):
        element_to_insert = array[i]
        position = i

        while position > 0 and array[position - 1] > element_to_insert:
            array[position] = array[position - 1]
            position -= 1

        array[position] = element_to_insert

    return array

array = [4,5,2,7,9,7,45,2,3,4,6,7,3,90]
print(insertion_sort(array))
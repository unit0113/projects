from turtle import position


def shell_sort(array):
    sublist_count = len(array) // 2
    while sublist_count > 0:
        for start in range(sublist_count):
            gap_insertion(array, start, sublist_count)
        
        sublist_count = sublist_count // 2

    return array


def gap_insertion(array, start, gap):
    for i in range(start+gap, len(array), gap):
        current_value = array[i]
        position = i

        while position >= gap and array[position-gap] > current_value:
            array[position] = array[position-gap]
            position = position - gap
        
        array[position] = current_value

    return array

array = [4,5,2,7,9,7,45,2,3,4,6,7,3,90]
print(shell_sort(array))
import numpy as np

def counting_sort(array):
    max_element = max(array)
    min_element = min(array)
    range_elements = max_element - min_element + 1

    count_array = [0 for _ in range(range_elements)]
    output_array = [0 for _ in array]

    # Count each occurence
    for number in array:
        count_array[number - min_element] += 1

    # Update count array so that is contains actual positions of this element in the output array
    for i in range(1, len(count_array)):
        count_array[i] += count_array[i-1]

    # Build output array
    for number in array:
        output_array[count_array[number - min_element] - 1] = number
        count_array[number - min_element] -= 1

    return output_array

array = np.random.randint(0,10,10)
array = counting_sort(array)
print(array)
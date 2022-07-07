import numpy as np

def counting_sort(array, exp):
    count_array = [0 for _ in range(10)]
    output_array = [0 for _ in array]

    # Count each occurence
    for number in array:
        index = number // exp
        count_array[index % 10] += 1

    # Update count array so that is contains actual positions of this element in the output array
    for i in range(1, 10):
        count_array[i] += count_array[i-1]

    # Build output array
    i = len(array) - 1
    while i >= 0:
        index = array[i] // exp
        output_array[count_array[index % 10] - 1] = array[i]
        count_array[index % 10] -= 1
        i -= 1

    # Copy output to original array
    for index, number in enumerate(output_array):
        array[index] = number


def radix_sort(array):
    max_element = max(array)
    exp = 1
    while max_element / exp > 1:
        counting_sort(array, exp)
        exp *= 10



array = np.random.randint(0,1000,100)
radix_sort(array)
print(array)
import enum


def count_long_subarray(A):
    '''
    Input:  A     | Python Tuple of positive integers
    Output: count | number of longest increasing subarrays of A
    '''
    count = 0
    max_length = 0
    cur_length = 0
    for index, item in enumerate(A[1:]):
        if item > A[index]:
            cur_length += 1
        else:
            cur_length = 0

        if cur_length > max_length:
            count = 1
            max_length = cur_length
        elif cur_length == max_length:
            count += 1

    return count

B = [2, 2, 4, 1, 4]
print(count_long_subarray(B))
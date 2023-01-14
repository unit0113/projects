import random


def merge_sort(array: list) -> None:
    width = 1
    n = len(array)
    
    while width < n:
        left = 0
        while left < n:
            right = min(left + (width * 2 - 1), n - 1)
            middle = min(left + width - 1, n - 1)
            _merge(array, left, middle, right)
            left += width * 2
        
        width *= 2
    
    return array


def _merge(array, left, middle, right):
    n_left = middle - left + 1
    n_right = right - middle

    L = [0] * n_left 
    R = [0] * n_right 
    for i in range(0, n_left): 
        L[i] = array[left + i] 
    for i in range(0, n_right): 
        R[i] = array[middle + i + 1] 
  
    i, j, k = 0, 0, left 
    while i < n_left and j < n_right: 
        if L[i] <= R[j]: 
            array[k] = L[i] 
            i += 1
        else: 
            array[k] = R[j] 
            j += 1
        k += 1
  
    while i < n_left: 
        array[k] = L[i] 
        i += 1
        k += 1
  
    while j < n_right: 
        array[k] = R[j] 
        j += 1
        k += 1


if __name__ == "__main__":
    array = random.sample(range(0, 51), 25)
    merge_sort(array)
    print(array)
    
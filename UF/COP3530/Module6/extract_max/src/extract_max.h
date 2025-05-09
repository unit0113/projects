/*
  Quiz 7: Extract Max

  Write C++ code for a function, extractMax() that takes as input 
  a max heap, arr represented as an integer array and the size of 
  the array, size. The function deletes the maximum element in the 
  max heap and returns the new max heap array after deletion.

  Sample Input:
    3
    9 8 7

  Sample Output:
    8 7
    
  Explanation:

  Input:  Line 1 denotes the number of elements, size in the 
            heap. Line 2 denotes the contents of the max heap 
            that is passed into extractMax() function.

  Output: Output is the max heap after deletion.
*/

#include <iostream>
#include <algorithm>

int* extractMax(int arr[], int size)
{

    // code here
    // main prints the final heap array from 0 to size - 2
    // you don't need to print the array but instead return
    arr[0] = arr[--size];

    // Initialize helper variables
    int index{};
    int left{};
    int right{};
    int largest{};

    // Heapify down
    while (index < size) {
        left = 2 * index + 1;
        right = 2 * index + 2;
        largest = index;
        
        // Find index of largest between parent and children
        if (left < size && arr[left] > arr[largest]) {
            largest = left;
        }
        if (right < size && arr[right] > arr[largest]) {
            largest = right;
        }
                                          
        if (largest == index) {
            break;
        }
        
        std::swap(arr[index], arr[largest]);
        index = largest;       
  }

    return arr; 
}

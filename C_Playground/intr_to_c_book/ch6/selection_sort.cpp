#include <iostream>
#include <vector>
#include <chrono>

void selection_sort(std::vector<int> &arr);
void selection_sort(int arr[], int size);

int main() {
    size_t size = 0;
    std::cout << "Enter size of array: ";
    std::cin >> size;

    auto start = std::chrono::high_resolution_clock::now();

    srand(time(NULL));

    std::vector<int> arr;
    for (size_t i = 0; i < size; i++) {
        arr.push_back(rand() % 10);
    }

    std::cout << "Original array:";
    for (int x: arr) {
        std::cout << ' ' << x;
    }
    std::cout << std::endl;

    selection_sort(arr);

    std::cout << "Sorted array:";
    for (int x: arr) {
        std::cout << ' ' << x;
    }
    std::cout << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;

    return 0;
}


void selection_sort(std::vector<int> &arr) {
    int temp = 0;
    int min = 10;
    int min_index = 0;
    for (size_t i = 0; i < arr.size(); i++) {
        min = arr[i];
        min_index = i;
        for (size_t j = i + 1; j < arr.size(); j++){
            if (arr[j] < min) {
                min = arr[j];
                min_index = j;
            }
        }

        if (min_index != i) {
            temp = arr[i];
            arr[i] = min;
            arr[min_index] = temp;
        }
        
    }
}


void selection_sort(int arr[], const int size) {
    int temp = 0;
    int min = 10;
    int min_index = 0;
    for (size_t i = 0; i < size; i++) {
        min = arr[i];
        min_index = i;
        for (size_t j = i + 1; j < size; j++){
            if (arr[j] < min) {
                min = arr[j];
                min_index = j;
            }
        }

        if (min_index != i) {
            temp = arr[i];
            arr[i] = min;
            arr[min_index] = temp;
        }
        
    }
}
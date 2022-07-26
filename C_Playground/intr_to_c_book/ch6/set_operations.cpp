#include <iostream>
#include <chrono>
#include <vector>

std::vector<int> get_union(int arr1[], int arr2[], int size);
std::vector<int> get_intersection(int arr1[], int arr2[], int size);
void selection_sort(int arr[], int size);

int main() {

    auto start = std::chrono::high_resolution_clock::now();

    srand(time(NULL));

    const int size = 10;

    int arr1[size];
    int arr2[size];

    for (size_t i = 0; i < size; i++) {
        arr1[i] = rand() % 20;
        arr2[i] = rand() % 20;
    }

    selection_sort(arr1, size);
    selection_sort(arr2, size);

    std::cout << "Original array1:";
    for (int x: arr1) {
        std::cout << ' ' << x;
    }
    std::cout << std::endl;

    std::cout << "Original array2:";
    for (int x: arr2) {
        std::cout << ' ' << x;
    }
    std::cout << std::endl;


    std::vector<int> union_vec = get_union(arr1, arr2, size);

    std::cout << "Union:";
    for (int x: union_vec) {
        std::cout << ' ' << x;
    }
    std::cout << std::endl;


    std::vector<int> intersection_vec = get_intersection(arr1, arr2, size);

    std::cout << "Intersection:";
    for (int x: intersection_vec) {
        std::cout << ' ' << x;
    }
    std::cout << std::endl;   
    

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;

    return 0;
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


std::vector<int> get_union(int arr1[], int arr2[], int size) {
    std::vector<int> return_vec;
    size_t i = 0;
    size_t j = 0;

    while (i < size && j < size) {
        if (arr1[i] < arr2[j]) {
            if (return_vec.size() == 0 || arr1[i] != return_vec.back()) {
                return_vec.push_back(arr1[i]);
            }
            i++;
        } else if (arr1[i] > arr2[j]) {
            if (return_vec.size() == 0 || arr2[j] != return_vec.back()) {
                return_vec.push_back(arr2[j]);
            }
            j++;
        } else {
            if (return_vec.size() == 0 || arr1[i] != return_vec.back()) {
                return_vec.push_back(arr1[i]);
            }
            i++;
            j++;
        }
    }

    while (i < size) {
        if (arr1[i] != return_vec.back()) {
                return_vec.push_back(arr1[i]);
        }
        i++;
    }

    while (j < size) {
        if (arr2[j] != return_vec.back()) {
                return_vec.push_back(arr2[j]);
        }
        j++;
    }

    return return_vec;
}


std::vector<int> get_intersection(int arr1[], int arr2[], int size) {
    std::vector<int> return_vec;
    size_t i = 0;
    size_t j = 0;

    while (i < size && j < size) {
        if (arr1[i] < arr2[j]) {
            i++;
        } else if (arr1[i] > arr2[j]) {
            j++;
        } else {
            if (return_vec.size() == 0 || arr1[i] != return_vec.back()) {
                return_vec.push_back(arr1[i]);
            }
            i++;
            j++;
        }
    }

    return return_vec;
}

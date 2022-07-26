#include <iostream>


int dot(int arr1[], int arr2[], int size);
int** matrix_mult(int arr1[][2], int arr2[][2], const int size);        // don't include first dimension, int** required to return 2d array
void print_arr(int** arr, int size);


int main() {

    int arr1[] = {1,2,3};
    int arr2[] = {4,5,6};
    int size = 3;

    std::cout << "Dot Product: " << dot(arr1, arr2, size) << std::endl;


    const int size2 = 2;
    int arr3[size2][size2] = {1, 2, 3, 4};
    int arr4[size2][size2] = {5, 6, 7, 8};

    int** result_matrix = matrix_mult(arr3, arr4, size2);
    print_arr(result_matrix, size2);

    return 0;
}


int dot(int arr1[], int arr2[], int size) {

    int sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += arr1[i] * arr2[i];
    }

    return sum;
}


int** matrix_mult(int arr1[][2], int arr2[][2], const int size) {

    int** result_matrix = 0;                //must initialize cell by cell
    result_matrix = new int * [size];
    
    for (size_t h = 0; h < size; h++) {
        result_matrix[h] = new int[size];

        for (size_t w = 0; w < size; w++) {
            result_matrix[h][w] = arr1[h][0] * arr2[0][w] + arr1[h][1] * arr2[1][w];
        }
    }

    return result_matrix;
}


void print_arr(int** arr, int size) {

    std::cout << "Matrix Multiplication:\n";
    for (size_t row = 0; row < size; row++) {
        for (size_t col = 0; col < size; col++) {
            std::cout << arr[row][col] << '\t';
        }
        std::cout << std::endl;
    }
}

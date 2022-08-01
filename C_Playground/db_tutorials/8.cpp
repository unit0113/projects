#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>


std::vector<int> gen_random_vec(int num_values, int min_val, int max_val);
void bubble_sort(std::vector<int> &arr);
void print_arr(std::vector<int> &arr);
void print_helper(int x);
int factorial(int num);
void print_1d_vec(std::vector<int> arr);


int main() {

    // Bubble sort
    std::vector<int> vec_values = gen_random_vec(10, 0, 100);
    print_arr(vec_values);
    bubble_sort(vec_values);

    // Factorial
    std::cout << "Factorial 5: " << factorial(5) << std::endl;

    // Print visual array
    std::vector<int> new_vec = gen_random_vec(15, 0, 20);
    print_1d_vec(new_vec);







    return 0;
}


std::vector<int> gen_random_vec(int num_values, int min_val, int max_val) {
    std::vector<int> vec_random;
    srand(time(NULL));
    int range = 1 + max_val - min_val;
    for (size_t i = 0; i < num_values; i++) {
        vec_random.push_back(rand() % range + min_val);
    }
    return vec_random;
}


void bubble_sort(std::vector<int> &arr) {
    int temp = 0;
    bool swap = false;
    for (size_t i = arr.size() - 1; i > 0; i--) {
        swap = false;
        for (size_t j = 1; j <= i; j++) {
            if (arr[j-1] > arr[j]) {
                temp = arr[j-1];
                arr[j-1] = arr[j];
                arr[j] = temp;
                swap = true;
            }
        }
        print_arr(arr);
        if (!swap) {
            break;
        }
    }
}


void print_arr(std::vector<int> &arr) {
    std::cout << arr[0];
    std::for_each(arr.begin() + 1, arr.end(), print_helper);
    std::cout << std::endl;
}


void print_helper(int x) {
    std::cout << ", " << x;
}


int factorial(int num) {
    if (num == 1) {
        return 1;
    } else {
        return num * factorial(num - 1);
    }
}


void print_1d_vec(std::vector<int> arr) {
    // Print top dashes
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << "------";
    }
    std::cout << '-' << std::endl;

    // Print indexes
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << "| " << std::setw(2) << i << "  ";
    }
    std::cout << '|' << std::endl;   

    // Print mid dashes
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << "------";
    }
    std::cout << '-' << std::endl; 

    // Print items
    for (int x: arr) {
        std::cout << "| " << std::setw(2) << x << "  ";
    }
    std::cout << '|' << std::endl;

    // Print bottom dashes
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << "------";
    }
    std::cout << '-' << std::endl;
}
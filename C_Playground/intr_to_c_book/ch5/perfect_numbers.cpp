#include <iostream>
#include <chrono>
#include <vector>
#include <string>


void print_if_perfect(int num);
std::vector<int> get_factors(int num);


int main() {

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 1; i <= 100000; i++) {
        print_if_perfect(i);
    }



    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;

    return 0;
}


void print_if_perfect(int num) {
    std::vector<int> factors = get_factors(num);
    int sum_factors = 0;
    for (int x: factors) {
        sum_factors += x;
    }

    if (sum_factors == num) {
        std::string print_string = std::to_string(num) + "; Factors: ";
        for (int x: factors) {
            print_string += std::to_string(x) + ", ";
        }

        std::cout << print_string.substr(0, print_string.size()-2) << std::endl;
    }
}


std::vector<int> get_factors(int num) {
    std::vector<int> factors;
    for (size_t i = 1; i <= num / 2; i++) {
        if ((num % i) == 0) {
            factors.push_back(i);
        }
    }

    return factors;
}

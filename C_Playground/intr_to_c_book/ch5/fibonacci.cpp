#include <iostream>
#include <chrono>
#include <array>


uint64_t fibonacci_rec(int n);
uint64_t fibonacci_iter(int n);
int64_t fibonacci_dynamic(std::array<int64_t, 100> &memo, int n);


int main() {

    int num = 0;

    std::cout << "Calculate which nth fibonacci number: ";
    std::cin >> num;

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Number, recursive: " << fibonacci_rec(num) << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;


    start = std::chrono::high_resolution_clock::now();

    std::cout << "Number, iterative: " << fibonacci_iter(num) << std::endl;

    end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;

    
    start = std::chrono::high_resolution_clock::now();

    std::array<int64_t, 100> memo;
    memo.fill(-1);
    memo[1] = 0;
    memo[2] = 1;

    std::cout << "Number, dynamic: " << fibonacci_dynamic(memo, num) << std::endl;

    end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;

    return 0;
}


uint64_t fibonacci_rec(int n) {
    if ((n == 1) || (n == 2)) {
        return (uint64_t)n - 1;
    } else {
        return fibonacci_rec(n-2) + fibonacci_rec(n-1);
    }
}


uint64_t fibonacci_iter(int n) {
    uint64_t sum = 0;
    uint64_t num1 = 0;
    uint64_t num2 = 1;
    for (size_t i = 2; i < n; i++) {
        sum = num1 + num2;
        num1 = num2;
        num2 = sum;
    }

    return sum;
}


int64_t fibonacci_dynamic(std::array<int64_t, 100> &memo, int n) {
    if (memo[n] == -1) {
        memo[n] = fibonacci_dynamic(memo, n-1) + fibonacci_dynamic(memo, n-2);
    }

    return memo[n];
}
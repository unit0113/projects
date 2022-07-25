#include <iostream>
#include <chrono>
#include <array>


uint64_t fibonacci_rec(int n);
uint64_t fibonacci_iter(int n);
uint64_t fibonacci_dynamic(int n);


int main() {

    int num = 0;

    std::cout << "Calculate which nth fibonacci number: ";
    std::cin >> num;

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Number, recursive: " << fibonacci_rec(num-1) << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;


    start = std::chrono::high_resolution_clock::now();

    std::cout << "Number, iterative: " << fibonacci_iter(num) << std::endl;

    end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;

    
    start = std::chrono::high_resolution_clock::now();

    std::cout << "Number, dynamic: " << fibonacci_dynamic(num-1) << std::endl;

    end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;



    return 0;
}


uint64_t fibonacci_rec(int n) {
    if ((n == 0) || (n == 1)) {
        return (uint64_t)n;
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


uint64_t fibonacci_dynamic(int n) {
    static std::array<uint64_t, 100> memo = {};
    memo.fill(-1);
    memo[0] = 0;
    memo[1] = 1;

    if (memo[n] == -1) {
        memo[n] = fibonacci_dynamic(n-1) + fibonacci_dynamic(n-2);
    }

    return memo[n];

}
#include <iostream>
#include <math.h>


int main() {
    int num = 0;
    std::cout << "Enter a number: ";
    std::cin >> num;

    bool prime = false;
    for (size_t i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) {
            prime = true;
            break;
        }
    }

    if (prime == false) {
        std::cout << num << " is prime!\n";
    } else {
        std::cout << num << " is not prime\n";
    }

    return 0;
}
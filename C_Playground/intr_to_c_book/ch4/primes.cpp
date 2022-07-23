#include <iostream>
#include <math.h>

bool check_prime(int num);

int main() {

    std::cout << "Primes from 1 to 100:";
    for (size_t i = 1; i <= 100; i++) {
        if (check_prime(i)) {
            std::cout << ' ' << i;
        }
    }
    
    std::cout << std::endl;

    return 0;
}


bool check_prime(int num) {
    bool prime = true;
    for (size_t i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) {
            prime = false;
            break;
        }
    }

    return prime;
}
#include <iostream>


int power(int base, int exponent);


int main() {

    int base = 0;
    int exponent = 0;

    std::cout << "Enter a base and an exponent: ";
    std::cin >> base >> exponent;

    std::cout << "Result: " << power(base, exponent) << std::endl;

    return 0;
}


int power(int base, int exponent) {
    if (exponent == 0) {
        return 1;
    } else {
        return base * power(base, exponent - 1);
    }
}

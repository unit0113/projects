#include <iostream>


int main() {

    int input = 0;
    std::cout << "Enter an integer: ";
    std::cin >> input;

    int factorial = 1;
    for (size_t i = 2; i <= input; i++) {
        factorial *= i;
    }

    std::cout << input << " factorial is: " << factorial << std::endl;

    return 0;
}
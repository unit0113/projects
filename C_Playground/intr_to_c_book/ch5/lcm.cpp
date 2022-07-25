#include <iostream>


int main() {

    int num1 = 0;
    int num2 = 0;
    std::cout << "Enter two integers: ";
    std::cin >> num1 >> num2;

    for (size_t i = 2; i <= std::min(num1, num2); i++) {
        if ((num1 % i == 0) && (num2 % i == 0)) {
            std::cout << "LCM: " << i << std::endl;
            break;
        }
    }

    return 0;
}
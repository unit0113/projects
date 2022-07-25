#include <iostream>


int main() {
    std::string s_input = "";
    std::cout << "Enter an integer: ";
    std::cin >> s_input;

    int sum = 0;
    for (char c: s_input) {
        sum += c - '0';
    }

    std::cout << "Sum of digits: " << sum << std::endl;

    return 0;
}

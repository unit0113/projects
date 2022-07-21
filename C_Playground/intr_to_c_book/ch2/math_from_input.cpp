#include <iostream>
#include <algorithm>



int main() {
    int num1 {};
    int num2 {};
    int num3 {};

    std::cout << "Enter three integers: ";
    std::cin >> num1 >> num2 >> num3;

    // Calc sum
    int sum = num1 + num2 + num3;
    std::cout << "Sum: " << sum << '\n';

    // Calc avg
    double avg = sum / 3.0;
    std::cout << "Avg: " << avg << '\n';

    // Product
    std::cout << "Product: " << num1 * num2 * num3 << '\n';

    //Smallest
    std::cout << "Minimum: " << std::min({num1, num2, num3}) << '\n';

    //Largest
    std::cout << "Maximum: " << std::max({num1, num2, num3}) << '\n';

    return 0;
}
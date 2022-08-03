#include <iostream>
#include <iomanip>

double f_to_c(double fahrenheit);


int main() {

    auto cell_width = std::setw(8);
    std::cout << std::setprecision(3) << std::fixed;

    std::cout << "   " << "F°" << "   |   " << "C°" << std::endl;
    std::cout << std::string(17, '-') << std::endl;

    for (double f = 0; f <= 212; f++) {
        std::cout << cell_width << std::noshowpos << f << '|' << cell_width << std::showpos << f_to_c(f) << std::endl;
    }

    return 0;
}


double f_to_c(double fahrenheit) {
    return (5.0 / 9.0) * (fahrenheit - 32.0);
}
#include <iostream>

enum Color {
    RED, GREEN, BLUE, PURPLE, ORANGE
};

int main() {
    std::cout << BLUE << std::endl;     // prints index of BLUE in Color enum

    Color mycolor = RED;
    std::cout << mycolor << std::endl;


    return 0;
}
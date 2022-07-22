#include <iostream>


int square(int x);
int cube(int x);
int quad(int x);

int main() {

    std::cout << "N\t" << "N\xc2\xb2\t" << "N\xc2\xb3\t" << "N\xe2\x81\xb4\n\n";

    for (size_t i = 1; i <= 10; i++) {
        std::cout << i << '\t' << square(i) << '\t' << cube(i) << '\t' << quad(i) << std::endl;
    }

    return 0;
}


int square(int x) {
    return x * x;
}


int cube(int x) {
    return x * x * x;
}


int quad(int x) {
    return x * x * x * x;
}
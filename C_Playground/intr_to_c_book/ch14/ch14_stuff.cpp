#include <iostream>
#include <cstdarg>


int sum(int size,...);


int main(int argc, char* argv[]) {

    std::cout << "Arguments:";
    for (size_t i = 0; i < argc; i++) {
        std::cout << ' ' << argv[i];
    }
    std::cout << std::endl;

    std::cout << sum(5, 4,5,2,6,7) << std::endl;
    std::cout << sum(3, 9, 0, 4) << std::endl;

    return 0;
}

int sum(int size,...) {
    va_list valist;
    int total = 0;
    va_start(valist, size);
    for (size_t i = 0; i < size; i++) {
        total += va_arg(valist, int);
    }
    va_end(valist);

    return total;

}
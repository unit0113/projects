#include <iostream>
#include <chrono>


int main() {

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t side1 = 1; side1 <= 500; side1++) {
        for (size_t side2 = side1; side2 <= 500; side2++) {
            for (size_t side3 = side2; side3 <= 500; side3++) {
                if (side3 * side3 == side1 * side1 + side2 * side2) {
                    std::cout << "Triple: " << side1 << ", " << side2 << ", " << side3 << std::endl;
                } else if (side3 * side3 > side1 * side1 + side2 * side2) {
                    break;
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;

    return 0;
}
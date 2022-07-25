#include <iostream>

bool flip();

int main() {

    int heads = 0;
    int tails = 0;
    for (size_t i = 0; i < 100; i++) {
        (flip()) ? heads++ : tails++;
    }

    std::cout << "Heads: " << heads << std::endl;
    std::cout << "Tails: " << tails << std::endl;

    return 0;
}


bool flip() {
    return rand() % 2;
}
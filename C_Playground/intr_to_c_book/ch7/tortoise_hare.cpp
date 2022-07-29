#include <iostream>
#include <Windows.h>


#define FINISHLINE 75


int move_tortoise();
int move_hare();
void print_status(int tortoise_pos, int hare_pos);


int main() {

    srand(time(NULL));

    std::cout << "3...";
    Sleep(1000);
    std::cout << "2...";
    Sleep(1000);
    std::cout << "1...";
    Sleep(1000);
    std::cout << "BANG!!\n";
    std::cout << "And they're off!\n";

    int tortoise_pos = 0;
    int hare_pos = 0;
    int cycles = 0;

    print_status(tortoise_pos, hare_pos);

    while (tortoise_pos < FINISHLINE && hare_pos < FINISHLINE) {
        tortoise_pos += move_tortoise();
        hare_pos += move_hare();
        print_status(tortoise_pos, hare_pos);
        cycles++;
        Sleep(100);
    }

    std::cout << std::endl;

    if (tortoise_pos == hare_pos) {
        std::cout << "It's a tie!\n";
    } else if (tortoise_pos > hare_pos) {
        std::cout << "The Tortoise wins!!\n";
    } else {
        std::cout << "The wabbit wins!!\n";
    }

    std::cout << "This race lasted " << cycles << " cycles" << std::endl;

    return 0;
}


void print_status(int tortoise_pos, int hare_pos) {
    std::cout << "\r                                                             ";
    std::cout << '\r' << "Tortoise: " << tortoise_pos << "\tHare: " << hare_pos;
}


int move_tortoise() {
    int rand_num = rand() % 10 + 1;
    if (rand_num < 6) {
        return 3;
    } else if (rand_num < 8) {
        return -6;
    } else {
        return 1;
    }
}


int move_hare() {
    int rand_num = rand() % 10 + 1;
    if (rand_num < 3) {
        return 0;
    } else if (rand_num < 5) {
        return 9;
    } else if (rand_num < 6) {
        return -12;
    } else if (rand_num < 9) {
        return 1;
    } else {
        return -2;
    }
}
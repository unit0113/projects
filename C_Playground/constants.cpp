#include <iostream>

int main() {
    const int A = 10;
    std::cout << A << std::endl;
    std::cout << A + 30 << std::endl;

    int b = A;
    std::cout << b << std::endl;
    std::cout << b + 40 << std::endl;

    const int CONSTANT = 20;        // all caps for consts

    int a = 10;
    // Cannot change value of thing ptr points to
    const int *p1 = &a;  //method one

    int d = 20;
    // can change where ptr points
    std::cout << p1 << std::endl;
    p1 = &d;
    std::cout << p1 << std::endl;


    // cannot change address, can change value
    int* const p2 = &a; //method two
    std::cout << *p2 << std::endl;
    *p2 = 50;
    std::cout << *p2 << std::endl;


    //Can't change anything
    const int* const p3 = &a;

    return 0;
}
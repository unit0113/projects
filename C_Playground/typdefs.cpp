#include <iostream>

//used for readability and making global changes
//to the type of commonly used variables
typedef unsigned int age_t;         //_t is convention

typedef unsigned char byte;

int main() {
    age_t myage = 10;
    std::cout << myage << std::endl;
    std::cout << myage + 30 << std::endl;

    byte b = 70;
    std::cout << b << std::endl;    //treated as char
    std::cout << unsigned(b) << std::endl;  // treats as number
    byte b2 = 20;
    std::cout << b + b2 << std::endl;       // also treated as number
    byte b3 = b + b2;
    std::cout << b3 << std::endl;           //char again


    return 0;
}
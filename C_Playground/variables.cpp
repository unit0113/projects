#include <iostream>
#include <limits.h>

int main() {

    // Numerical
    int x = 10;
    short int a = 10;
    std::cout << "Integer (int): " << sizeof(int) << std::endl;
    std::cout << "Int max: " << INT_MAX << std::endl;
    std::cout << "Int min: " << INT_MIN << std::endl;
    std::cout << "Unsigned int max: " << UINT_MAX << std::endl;
    std::cout << "Short (short): " << sizeof(short) << std::endl;
    std::cout << "Short max: " << SHRT_MAX<< std::endl;
    std::cout << "Unisgned short max: " << USHRT_MAX<< std::endl;
    std::cout << "Long (long): " << sizeof(long) << std::endl;
    std::cout << "Long long (long): " << sizeof(long long) << std::endl;
    std::cout << "Long long max: " << LLONG_MAX << std::endl;

    std::cout << "Float (float): " << sizeof(float) << std::endl; // 7 digits
    std::cout << "Double (double): " << sizeof(double) << std::endl; // 15 digits

    // Textual
    char c = 'a';
    std::cout << "Char (char): " << sizeof(char) << std::endl;
    std::string s = "Hello World!";
    std::cout << "String (str): " << sizeof(std::string) << std::endl;

    // Boolean
    bool b = true;
    std::cout << b << std::endl;
    std::cout << "Bool (bool): " << sizeof(bool) << std::endl;

    return 0;
}
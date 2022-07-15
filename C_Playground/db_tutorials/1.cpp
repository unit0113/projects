#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

int main(int argc, char** argv) {

    if(argc != 1) {
        std::cout << "You entered " << argc << " arguments\n";
    }

    for(int i = 0;  i < argc; ++i) {
        std::cout << argv[i] << std::endl;
    }

    std::string question { "Enter Number 1: " };
    std::string num1, num2;
    std::cout << question;
    getline(std::cin, num1);
    std::cout << "Enter Number 2: ";
    getline(std::cin, num2);

    int nNum1 = std::stoi(num1);
    int nNum2 = std::stoi(num2);

    printf("%d + %d = %d\n", nNum1, nNum2, nNum1 + nNum2);

    return 0;
}
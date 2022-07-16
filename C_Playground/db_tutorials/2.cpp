#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <limits>

void print_important();

int main() {

    std::string sAge;
    std::cout << "Enter your age: ";
    getline(std::cin, sAge);
    int nAge = std::stoi(sAge);

    if ((nAge >= 1) && (nAge <= 18)) {
        print_important();
    } else if ((nAge == 21) || (nAge == 50)) {
        print_important();
    } else if (nAge >= 65) {
        print_important();
    } else {
        std::cout << "Not an important birthday!" << std::endl;
    }


    const int KINDERGARTEN_AGE = 5;
    std::string sAge2;
    std::cout << "Enter your age: ";
    getline(std::cin, sAge2);
    int nAge2 = std::stoi(sAge2);

    if (nAge2 < KINDERGARTEN_AGE) {
        std::cout << "Too young for school" << std::endl;
    } else if (nAge2 == 5) {
        std::cout << "Go to Kindergarten" << std::endl;
    } else if ((nAge2 > KINDERGARTEN_AGE) && (nAge2 <= 17)) {
        std::cout << "Grade " << nAge2 - KINDERGARTEN_AGE << std::endl;
    } else {
        std::cout << "Go to college" << std::endl;
    }


    // arrays
    int arrnNums[10] = {1};
    int arrnNums2[] = {1, 2, 3};
    int arrnNums3[5] = {8, 9};

    std::cout << "Array Size: " << sizeof(arrnNums2) / sizeof(*arrnNums2) << std::endl;

    int arrnNums3d[2][2][2] = {{{1, 2}, {3, 4}},
        {{5, 6}, {7, 8}}};

    std::cout << arrnNums3d[1][1][1] << std::endl;


    // vectors (lists?)
    std::vector<int> vecnRandNums(2);
    vecnRandNums[0] = 10;
    vecnRandNums[1] = 20;
    vecnRandNums.push_back(30);
    std::cout << "Last: " << vecnRandNums[vecnRandNums.size() - 1] << std::endl;

    std::string sSent = "Random string of random words";
    std::vector<std::string> vecsWords;
    std::stringstream ss(sSent);
    std::string sIndivStr;              // temp holding cell for while loop
    char cSpace = ' ';

    while(getline(ss, sIndivStr, cSpace)) {
        vecsWords.push_back(sIndivStr);
    }

    for (int i = 0; i < vecsWords.size(); i++) {
        std::cout << vecsWords[i] << std::endl;
    }


    // Calculator
    std::string sCalcInput = "";
    std::cout << "Enter calculation: ";
    getline(std::cin, sCalcInput);

    std::vector<std::string> vecsCalc;
    std::stringstream ss2(sCalcInput);
    std::string sIndivStr2;
    char cSpace2 = ' ';

    while(getline(ss2, sIndivStr2, cSpace2)) {
        vecsCalc.push_back(sIndivStr2);
    }

    std::string operation = vecsCalc[1];

    if (operation == "+") {
        std::cout << vecsCalc[0] << " + " << vecsCalc[2] << " = " << stoi(vecsCalc[0]) + stoi(vecsCalc[2]) << std::endl;
    } else if (operation == "-") {
        std::cout << vecsCalc[0] << " - " << vecsCalc[2] << " = " << stoi(vecsCalc[0]) - stoi(vecsCalc[2]) << std::endl;
    } else if (operation == "*") {
        std::cout << vecsCalc[0] << " * " << vecsCalc[2] << " = " << stoi(vecsCalc[0]) * stoi(vecsCalc[2]) << std::endl;
    } else if (operation == "/") {
        std::cout << vecsCalc[0] << " / " << vecsCalc[2] << " = " << stoi(vecsCalc[0]) / stoi(vecsCalc[2]) << std::endl;
    } else {
        std::cout << "Please enter only +, -, *, or /" << std::endl;
    }



    return 0;
}

void print_important() {
    std::cout << "Important birthday!" << std::endl;
}
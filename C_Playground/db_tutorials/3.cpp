#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#define NEWSECTION "****************************************"

void new_section();
double add_numbers(double, double);
void assign_age(int);
void assign_age2(int*);
std::vector<int> range(int, int, int);
double calc_interest(double, double, int);

int main() {

    // Fill vector with range starting at 0
    std::vector<int> myVec(10);
    std::iota(std::begin(myVec), std::end(myVec), 0);

    for (int i=0; i < myVec.size(); i++) {
        std::cout << myVec[i] << std::endl;
    }

    // for each loop
    for (auto y: myVec) std::cout << y << std::endl;

    new_section();

    // print even in range 1-10
    std::vector<int> myVec2(10);
    std::iota(std::begin(myVec2), std::end(myVec2), 1);
    for (auto val: myVec2) {
        if ((val % 2) == 0) {
            std::cout << val << std::endl;
        }
    }

    new_section();

    double num1 {};
    double num2 {};

    std::cout << "Enter Num 1: ";
    std::cin >> num1;
    std::cout << "Enter Num 2: ";
    std:: cin >> num2;
    printf("%.1f + %.1f = %.1f\n", num1, num2, add_numbers(num1, num2));

    new_section();

    // ints passed by value
    int age = 30;
    assign_age(age);
    std::cout << "Age is: " << age << std::endl;

    //pass by reference
    assign_age2(&age);
    std::cout << "Age is: " << age << std::endl;

    // Array with ptrs
    int intArray[] = {1, 2, 3, 4};
    int* pIntArray = intArray;

    new_section();

    std::cout << "1st: " << *pIntArray << ", Address: " << pIntArray << std::endl;
    pIntArray++;
    std::cout << "2nd: " << *pIntArray << ", Address: " << pIntArray++ << std::endl;
    std::cout << "3rd: " << *pIntArray << ", Address: " << pIntArray-- << std::endl;
    std::cout << "2nd again: " << *pIntArray << ", Address: " << pIntArray << std::endl;

    new_section();

    std::vector<int> myVec3 = range(1, 20, 2);
    for (auto y: myVec3) std::cout << y << std::endl;

    new_section();

    double value = calc_interest(1000, 8, 10);
    std::cout << value << std::endl;




    return 0;
}


void new_section() {
    std::cout << NEWSECTION << std::endl;
}

double add_numbers(double num1, double num2) {
    return num1 + num2;
}

void assign_age(int age) {
    age = 21;
}

void assign_age2(int* age) {
    *age = 21;
}

std::vector<int> range(int start, int end, int step) {
    std::vector<int> returnVec;
    for (int i = start; i < end; i += step) {
        returnVec.push_back(i);
    }

    return returnVec;
}

double calc_interest(double starting_val, double int_rate, int years) {
    double final_val = starting_val;
    double perc_int_rate = 1 + (int_rate / 100);
    for (int i = 0; i < years; i++) {
        final_val *= perc_int_rate;
    }

    return final_val;
}

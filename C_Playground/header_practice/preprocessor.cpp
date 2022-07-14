#include <iostream>

#define PI (3.14159)
#define square(a) a * a
#define TRIGGER
#define COND1
#define COND2   //contradictory to COND1 (both should not be true at same time)



int main() {

    std::cout << PI << std::endl;
    std::cout << square(5) << std::endl;

    int i { 5 };
    std::cout << square(i++) << std::endl;  //should be 25, then increments i
    std::cout << i << std::endl;            //should be 6, but gets incrememnted twice in square macro

    #undef PI       //deletes definition of PI

    #ifdef TRIGGER
    std::cout << "Trigger is defined!" << std::endl;
    #endif

    #undef TRIGGER
    #ifndef TRIGGER
    std::cout << "Trigger is not defined!" << std::endl;
    #endif

    #ifdef __linux__
        std::cout << "Linux machine" << std::endl;
    #endif

    #ifdef _WIN64
        std::cout << "Windows machine" << std::endl;
    #endif

    #ifdef COND1
    #ifdef COND2
    //#error            //raises error
    #endif
    #endif

    return 0;
}
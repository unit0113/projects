#include <iostream>
#include <string.h>

int main() {
    std::string mystring = "Hello, World!";
    std::cout << mystring.length() << std::endl;
    std::cout << mystring.at(4) << std::endl;   // pull individual index

    std::string mystring2 = "Hello, Leute!";
    if(mystring == mystring2) {
        std::cout << "Same" << std::endl;
    } else {
        std::cout << "Not the same" << std::endl;
    }

    std::string combined = mystring + ' ' + mystring2;
    std::cout << combined << std::endl;

    char text[10] = "Hello"; // C-style
    std::cout << text << std::endl;

    char text2[4];      // need 4 to store 3 (len+1)
    text2[0] = 'H';
    text2[1] = 'e';
    text2[2] = 'y';
    text2[3] = '\0';       // null terminator
    std::cout << text2 << std::endl;
    std::cout << strlen(text2) << std::endl;


    return 0;
}
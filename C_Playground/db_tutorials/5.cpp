#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>


#define NEWSECTION "****************************************"


void new_section();
std::string string_to_ascii_string(std::string);
std::string ascii_string_to_string(std::string);


int main() {

    // C version of strings
    char cString[] {'A', ' ', 'S', 't', 'r', 'i', 'n', 'g', '\0'};
    std::cout << "Array size: " << sizeof(cString) << std::endl;
    std::cout << cString << std::endl;

    new_section();

    // C++ version
    std::vector<std::string> strVec(10);
    std::string str("I'm a string");
    strVec[0] = str;
    
    // All ways to get first(slice for first two)
    std::cout << str[0] << std::endl;
    std::cout << str.at(0) << std::endl;
    std::cout << str.front() << std::endl;

    std::cout << str.back() << std::endl;
    std::cout << "Length: " << str.length() << std::endl;   // does not count null char

    new_section();

    //str copy
    std::string str2(str);
    strVec[1] = str2;
    std::string str3(str, 4);       //copy everything after x chars
    strVec[2] = str3;
    std::string str4(5, 'x');       //repeat x 5 times
    strVec[3] = str4;
    strVec[4] = str.append(" and you're not");      //does not affect str in strVec[0], does affect str

    str += " and you're not";
    std::cout << str << std::endl;

    str.append(str, 5, 10);         //starting at index five, get next 10 chars
    strVec[5] = str;
    str.erase(12, str.length() - 1);    //erase starting at index 12 to the end
    strVec[6] = str;

    // Search for substring
    if (str.find("string") != std::string::npos) {      //npos = no position, check if "string" in str
        std::cout << "1st index: " << str.find("string") << std::endl;
    }

    //return substring based on index, len
    std::cout << "Substring: " << str.substr(6, 6) << std::endl;

    //reverse
    reverse(str.begin(), str.end());
    std::cout << "Reversed: " << str << std::endl;

    //case transform
    transform(str2.begin(), str2.end(), str2.begin(), ::toupper);
    std::cout << "Upper: " << str2 << std::endl;
    transform(str2.begin(), str2.end(), str2.begin(), ::tolower);
    std::cout << "Lower again: " << str2 << std::endl;

    new_section();
    
    for (auto x: strVec) {
        if (x != ""){
            std::cout << x << std::endl;
        }
    }

    new_section();

    //ascii
    char aChar = 'A';
    int aInt = aChar;
    std::cout << "ASCII for A: " << aInt << std::endl;

    //From ascii to char
    std::string strNum = std::to_string(65 + 10);
    std::cout << "String of 65 + 10: " << strNum << std::endl;
    
    char kChar = aInt + 10;
    std::cout << "ASCII of 65 + 10: " << kChar << std::endl;

    new_section();

    std::string inputStr = "ELEPHANT";
    std::string asciiStr = string_to_ascii_string(inputStr);
    std::cout << asciiStr << std::endl;
    std::cout << ascii_string_to_string(asciiStr) << std::endl;

    std::cout << "Enter input string: ";
    std::cin >> inputStr;
    asciiStr = string_to_ascii_string(inputStr);
    std::cout << asciiStr << std::endl;
    std::cout << ascii_string_to_string(asciiStr) << std::endl;

    new_section();

    std::cout << "abs(-10) = " << std::abs(-10) << "\n";
    
    std::cout << "max(5,4) = " << std::max(5,4) << "\n";
    
    std::cout << "min(5,4) = " << std::min(5,4) << "\n";
    
    std::cout << "fmax(5.3,4.3) = " << std::fmax(5.3,4.3) << "\n";
    
    std::cout << "fmin(5.3,4.3) = " << std::fmin(5.3,4.3) << "\n";
    
    // e ^ x
    std::cout << "exp(1) = " << std::exp(1) << "\n";
    
    // 2 ^ x
    std::cout << "exp2(1) = " << std::exp2(1) << "\n";
    
    // e * e * e ~= 20 so log(20.079) ~= 3
    std::cout << "log(20.079) = " << std::log(20.079) << "\n";
    
    // 10 * 10 * 10 = 1000, so log10(1000) = 3
    std::cout << "log10(1000) = " << std::log10(1000) 
            << "\n";
    
    // 2 * 2 * 2 = 8
    std::cout << "log2(8) = " << std::log2(8) << "\n";
    
    // 2 ^ 3
    std::cout << "pow(2,3) = " << std::pow(2,3) << "\n";
    
    // Returns what times itself equals the provided value
    std::cout << "sqrt(100) = " << std::sqrt(100) << "\n";
    
    // What cubed equals the provided
    std::cout << "cbrt(1000) = " << std::cbrt(1000) << "\n";
    
    // Hypotenuse : SQRT(A^2 + B^2)
    std::cout << "hypot(2,3) = " << std::hypot(2,3) << "\n";
    
    std::cout << "ceil(10.45) = " << std::ceil(10.45) << "\n";
    
    std::cout << "floor(10.45) = " << std::floor(10.45) << "\n";
    
    std::cout << "round(10.45) = " << std::round(10.45) << "\n";
    
    // Also sin, cos, tan, asin, acos, atan, atan2,
    // sinh, cosh, tanh, asinh, acosh, atanh







    return 0;
}


void new_section() {
    std::cout << NEWSECTION << std::endl;
}


std::string string_to_ascii_string(std::string str) {
    std::string resultStr = "";
    for (char c: str) {
        resultStr += std::to_string((int)c);
    }

    return resultStr;
}


std::string ascii_string_to_string(std::string str) {
    std::string resultStr = "";
    for (int i = 0; i < str.length(); i += 2) {
        std::string sCharCode = "";
        sCharCode += str[i];
        sCharCode += str[i + 1];
        if (str[i] == '1') {
            sCharCode += str[i++ + 2];
        }

        int nCharCode = std::stoi(sCharCode);
        char cCharCode = nCharCode;
        resultStr += cCharCode;
        
    }

    return resultStr;
}
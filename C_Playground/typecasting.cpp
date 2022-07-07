#include <iostream>

using std::cout, std::endl, std::string;

int main() {

    //string s = "1023";
    //int i = std::stoi(s);
    //cout << i + 100 << endl;


    //float f = 10.1234;
    //cout << (int) f + 10 << endl;


    int i = 10;
    string s1 = "This number is: ";
    string s2 = std::to_string(i);

    cout << s1 + s2 << endl;

    return 0;
}
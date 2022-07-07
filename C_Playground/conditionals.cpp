#include <iostream>

using namespace std;

int main() {

    int a { 3 };
    if (a > 5) {
        cout << "Your number is larger than 5" << endl;
    } else if (a < 0){
        cout << "Your number is negative" << endl;
    } else {
        cout << "Your number is a baby" << endl;
    }


    return 0;
}
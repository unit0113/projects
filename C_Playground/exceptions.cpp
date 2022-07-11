#include <iostream>

using namespace std;

float divide(float, float);

int main() {

    float f1 = 20;
    float f2 = 0;

    try {
        cout << divide(f1, f2) << endl;
    } catch (int e) {
        if (e == 15) {
            cout << "Division by 0 error" << endl;
        } else {
            cerr << "Error" << endl;
        }
    }


    cout << "Still runs!" << endl;

    return 0;
}


float divide(float f1, float f2) {
    if (f2 == 0) {
        throw 15;
    } else {
        return f1 / f2;
    }
}
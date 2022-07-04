#include <iostream>


using namespace std;

int main() {


    int i { 0 };
    while (++i < 5) {
        cout << i << endl;
    }

    while (i > 0) {
        cout << i-- << endl;
    }

    do {
        cout << ++i << endl;
    } while (i < 5);


    for (int j = 0; j < 5; j++) {
        cout << j << endl;
    }

    return 0;
}
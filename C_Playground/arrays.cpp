#include <iostream>

using namespace std;

int main() {
    int myarray[20] = {10, 20, 30, 40};     // initilizes the first four, the rest are 0
    int myarray_2[20] = {};     // initializes all elements to 0
    int myarray_3[20];          // no initilization, all random values

    for (int i = 0; i < 20; i++){
        cout << myarray[i] << endl;
    }




    return 0;
}
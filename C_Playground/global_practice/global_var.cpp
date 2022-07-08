#include <iostream>
#include "extern_var.h"

using namespace std;

void func();

int x { 100 }; // add static keyword at the beginning to prevent other files from knowing about global var

int main() {
    func();
    cout << x << endl;

    //extern
    cout << add_to_x(10) << endl;



    return 0;
}

void func() {
    cout << x++ << endl;
}
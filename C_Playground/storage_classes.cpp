#include <iostream>


// extern int e;

int main() {
    static int a = 10;



    return 0;
}


int do_something() {
    // b doesn't get reallocated/assigned everytime do_something is called
    // save time for repeat calls for functions with constants
    static int b = 10;

    return 0;
}
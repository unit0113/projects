#include <iostream>
#include <stdio.h>
#include <string.h>


using namespace std;

int main() {
    char *str = (char *)malloc(20);
    strcpy(str, "Hello");
    cout << str << endl;

    return 0;
}

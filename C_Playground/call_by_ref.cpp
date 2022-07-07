#include <iostream>

using namespace std;

void pass_by_value(int);
void pass_by_ref(int &);

int main() {
    int a { 10 };
    cout << "a in main: " << a << endl;
    pass_by_value(a);
    cout << "a after pass_by_value: " << a << endl;
    pass_by_ref(a);
    cout << "a after pass_by_ref: " << a << endl;

    // Aliases (same variable/chunk of memory, two different names)
    int i1 { 10 };
    int &int1 { i1 };

    cout << i1 << endl;
    cout << int1 << endl;

    int1 += 90;

    cout << i1 << endl;


    return 0;
}

void pass_by_value(int x) {
    cout << "a in pass_by_value: " << ++x << endl;
}

void pass_by_ref(int &x) {
    cout << "a in pass_by_ref: " << ++x << endl;
}
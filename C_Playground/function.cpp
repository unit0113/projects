#include <iostream>

using namespace std;

int add(int, int);
double add(double, double);
float add(float, float);
string add(string, string);
int default_param_func(int, int, int=90);
void func();

int main() {
    cout << add(10, 20) << endl;
    cout << add(20.0, 30.0) << endl;
    cout << add(5.0f, 10.0f) << endl;
    cout << add("Hello ", "World!") << endl;
    cout << default_param_func(10, 20) << endl;
    func();
    func();
    func();
    func();
    func();
    func();

    return 0;
}

int add(int x, int y) {
    return x + y;
}

float add(float x, float y) {
    return x + y;
}

double add(double x, double y) {
    return x + y;
}

string add(string a, string b) {
    return a + b;
}

int default_param_func(int x, int y, int z) {
    return x + y + z;
}

void func() {
    static int x { 0 };    // static  means that x does get reinitialized on every func call
    // do stuff
    cout << x++ << endl;
}
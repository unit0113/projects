#include <iostream>
#include <functional>
#include <vector>
#include <ctime>
using namespace std;

double multby2(double num) {
    return num * 2;
};

double doMath(function<double(double)> func, double num) {
    return func(num);
};

double multby3(double num) {
    return num * 3;
};

bool isOdd(int num) {
    return (num % 2 == 1);
};

vector<int> chgList(vector<int> lst, function<bool(int)> func) {
    vector<int> retLst;
    for (auto& x: lst) {
        if (func(x)) {
            retLst.push_back(x);
        }
    }
    return retLst;
}

int main() {
    auto times2 = multby2;
    cout << "5 * 2 = " << times2(5) << endl;

    cout << "6 * 2 = " << doMath(times2, 6) << endl;

    vector<function<double(double)>> funcs(2);
    funcs[0] = multby2;
    funcs[1] = multby3;

    cout << "10 * 2 = " << funcs[0](10) << endl;
    cout << "10 * 3 = " << funcs[1](10) << endl;

    cout << "**************************************\n" << endl;

    vector<int> lst {1, 2, 3, 4, 5};
    vector<int> oddLst = chgList(lst, isOdd);
    for (auto& x: oddLst) {
        cout << x << endl;
    }

    cout << "**************************************\n" << endl;

    srand(time(NULL));
    int heads {};
    for (size_t i = 0; i < 100; i++) {
        if (rand() % 2 == 0) {
            heads++;
        }
    }
    cout << "Heads: " << heads << endl;
    cout << "Tails: " << 100 - heads << endl;



}
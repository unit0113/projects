#include <iostream>

using namespace std;

int main() {

    float n1, n2;
    char op;

    cout << "Enter the first number: ";
    cin >> n1;
    cout << "Choose the arithmetic operation: ";
    cin >> op;
    cout << "Enter the second number: ";
    cin >> n2;

    switch(op){

        case '+':
            cout << n1 + n2 << endl;
            break;

        case '-':
            cout << n1 - n2 << endl;
            break;

        case '*':
            cout << n1 * n2 << endl;
            break;

        case '/':
            cout << n1 / n2 << endl;
            break;

        case '%':
            cout << (int) n1 % (int) n2 << endl;
            break;

        default:
        cout << "Operation not supported!" << endl;
        break;
    }

    return 0;
}

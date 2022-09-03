#include <iostream>
#include <time.h> 
using namespace std;


int main() {
    cout << "Welcome to the greatest calculator on Earth!" << endl;
    srand(time(NULL));

    char run = 'y';
    char ch_opperand;
    int num1, num2, opperand, answer, cor_answer;
    while (run == 'y') {
        num1 = rand() % 200;
        num2 = rand() % 200;
        opperand = rand() % 3;

        if (opperand == 0) {
            cor_answer = num1 + num2;
            ch_opperand = '+';
        } else if (opperand == 1) {
            cor_answer = num1 - num2;
            ch_opperand = '-';
        } else {
            cor_answer = num1 * num2;
            ch_opperand = '*';
        }

        cout << "What's the result of " << num1 << ' ' << ch_opperand << ' ' << num2 << ": ";
        cin >> answer;
        if (answer == cor_answer) {
            cout << "Congrats! You got the result " << cor_answer << " right!" << endl;
        } else {
            cout << "Naah! The correct result is " << cor_answer << endl;
        }
        cout << "\nDo you want me to try again? (y|n): ";
        cin >> run;
    }

    cout << "See you later!" << endl;
}

#include <iostream>
#include <iomanip>
#include <array>
using namespace std;

auto COL_WIDTH = setw(7);


void printMonthHeader(int monthIndex);
void printDays(int day, int daysInMonth);


int main() {

    cout << "Enter a year: ";
    int year;
    cin >> year;

    cout << "Enter the first weekday of the year (0 for Sunday, 6 for Saturday): ";
    int day;
    cin >> day;


    array<int, 12> daysInMonth = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (year % 4 == 0) {
        daysInMonth[1] = 29;
    }

    cout << "Calender for " << year << endl;
    for (size_t i = 0; i < 12; i++) {
        printMonthHeader(i);
        printDays(day, daysInMonth[i]);
        day += daysInMonth[i];
    }

}


void printMonthHeader(int monthIndex) {
    static const array<string, 12> monthsInYear = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"};
    static const array<string, 7> daysInWeek = {"Sun", "Mon", "Tues", "Wed", "Thu", "Fri", "Sat"};
    cout << "-- " << monthsInYear[monthIndex] << " --" << endl;
    for (auto day: daysInWeek) {
        cout << right << COL_WIDTH << day;
    }
    cout << endl;
}


void printDays(int day, int daysInMonth) {
    for (size_t i = 0; i < day % 7; i++) {
        cout << COL_WIDTH << ' ';
    }

    for (size_t i = 1; i <= daysInMonth; i++) {
        cout << right << COL_WIDTH << i;
        if ((day + i) % 7 == 0) {
            cout << endl;
        }
    }
    cout << endl << endl;
}

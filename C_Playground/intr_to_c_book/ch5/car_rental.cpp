#include <iostream>
#include <vector>
#include <cmath>


#define MIN_FEE 25.0
#define PER_HR 5.0
#define MAX_PER_DAY 50.0
#define TAX 0.5
#define DAILY_HR_THRESH 24.0
#define HOURLY_HR_THRESH 8


void print_charges(std::vector<int> hours_vec);


int main() {

    std::vector<int> hours;
    int hour = 0;
    int car_num = 1;

    // Get length of rentals in hours from user
    while (true) {
        std::cout << "Enter hours for car " << car_num++ << ": ";
        std::cin >> hour;
        if (hour == -1) {
            break;
        }
        hours.push_back(hour);
    }

    // Print charges based on hours
    print_charges(hours);

    return 0;
}


void print_charges(std::vector<int> hours_vec) {
    std::cout << "Car\tHours\tCharge\n";

    double total_charge = 0;
    int total_hours = 0;
    double charge = 0;
    int car_num = 1;

    for (int x: hours_vec){
        if (x >= DAILY_HR_THRESH) {
            charge = std::ceil(x / DAILY_HR_THRESH) * MAX_PER_DAY;
        } else if (x > HOURLY_HR_THRESH) {
            charge = std::min(MAX_PER_DAY, MIN_FEE + PER_HR * (x - HOURLY_HR_THRESH));
        } else {
            charge = MIN_FEE;
        }
        
        charge += x * TAX;
        total_charge += charge;
        total_hours += x;
        std::cout << car_num++ << '\t' << x << '\t' << charge << std::endl;
    }

    std::cout << "TOTAL\t" << total_hours << '\t' << total_charge << std::endl;

}
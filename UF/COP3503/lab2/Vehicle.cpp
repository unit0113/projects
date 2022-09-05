#include "Vehicle.h"
using namespace std;


Vehicle::Vehicle(string make_, string model_, int year_, float price_, int mileage_) {
    make = make_;
    model = model_;
    year = year_;
    price = price_;
    mileage = mileage_;
}

void Vehicle::Display() const {
    cout << year << ' ' << make << ' ' << model << " $" << price << ' ' << mileage << endl;
}

string Vehicle::GetYearMakeModel() const {
    return to_string(year) + ' ' + make + ' ' + model;
}

float Vehicle::GetPrice() const {
    return price;
}
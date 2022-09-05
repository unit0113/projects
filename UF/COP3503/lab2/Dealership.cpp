#include "Dealership.h"
using namespace std;


Dealership::Dealership(string name_, unsigned int capacity_) {
    name = name_;
    capacity = capacity_;
    vector<Showroom> showrooms;
}

void Dealership::AddShowroom(Showroom s) {
    if (showrooms.size() == capacity) {
        cout << "Dealership is full, can't add another showroom!" << endl;
    } else {
        showrooms.push_back(s);
    }
}

float Dealership::GetAveragePrice() const {
    float sum_price { 0 };
    size_t num_cars { 0 };
    for (auto& room: showrooms) {
        for (auto& car: room.GetVehicleList()) {
            sum_price += car.GetPrice();
            num_cars++;
        }
    }
    return num_cars != 0 ? sum_price / num_cars : 0.0f;
}

void Dealership::ShowInventory() const {
    if (showrooms.size() != 0) {
        for (auto& room: showrooms) {
            room.ShowInventory();
            cout << endl;
        }
    } else {
        cout << name << " is empty!" << endl;
    }
    cout << "Average car price: $" << fixed << setprecision(2) << Dealership::GetAveragePrice();
}
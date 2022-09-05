#include "Showroom.h"
using namespace std;


Showroom::Showroom(string name_, unsigned int capacity_) {
    name = name_;
    capacity = capacity_;
    vector<Vehicle> vehicles;
}

vector<Vehicle> Showroom::GetVehicleList() const {
    return vehicles;
}

void Showroom::AddVehicle(Vehicle v) {
    if (vehicles.size() == capacity) {
        cout << "Showroom is full! Cannot add " << v.GetYearMakeModel() << endl;
    } else {
        vehicles.push_back(v);
    }
}

void Showroom::ShowInventory() const {
    if (vehicles.size() != 0) {
        cout << "Vehicles in " << name << endl;
        for (auto& car: vehicles) {
            car.Display();
        }
    } else {
        cout << name << " is empty!" << endl;
    }
    
}

float Showroom::GetInventoryValue() const {
    float sum = 0;
    for (auto& car: vehicles) {
        sum += car.GetPrice();
    }
    return sum;
}
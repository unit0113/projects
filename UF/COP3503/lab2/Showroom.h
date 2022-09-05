#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "Vehicle.h"
using namespace std;

class Showroom {
private:
    string name;
    unsigned int capacity;
    vector<Vehicle> vehicles;

public:
    Showroom(string name = "Unnamed Showroom", unsigned int capacity = 0);
    vector<Vehicle> GetVehicleList() const; 
    void AddVehicle(Vehicle v); 
    void ShowInventory() const; 
    float GetInventoryValue() const;
};

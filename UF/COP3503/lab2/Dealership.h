#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "Showroom.h"
#include "Vehicle.h"
using namespace std;


class Dealership {
private:
    string name;
    unsigned int capacity;
    vector<Showroom> showrooms;

public:
    Dealership(string name = "Generic Dealership", unsigned int capacity = 0);
    void AddShowroom(Showroom s); 
    float GetAveragePrice() const;
    void ShowInventory() const; 
};
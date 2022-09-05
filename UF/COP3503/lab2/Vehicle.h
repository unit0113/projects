#pragma once
#include <string>
#include <iostream>
using namespace std;


class Vehicle {
private:
    string make;
    string model;
    unsigned int year;
    float price;
    unsigned int mileage;

public:
    Vehicle(string make="COP3503", string model="Rust Bucket", int year=1900, float price=0.0f, int mileage=0); 
    void Display() const;
    string GetYearMakeModel() const; 
    float GetPrice() const; 
};

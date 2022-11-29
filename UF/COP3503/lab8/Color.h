#pragma once
#include <string>
#include <array>

class Color {
    int m_value;
    std::string m_name;
    static std::array<std::string, 16> hexVals;

    std::string hexValHelper(unsigned char val) const;

public:
    Color(int value, std::string name);
    void SetValue(int value); 
    void SetName(const char* name); 

    // Accessors 
    unsigned char GetR() const; 
    unsigned char GetG() const; 
    unsigned char GetB() const; 
    std::string GetHexValue() const; 
    std::string GetName() const {return m_name;}; 
};
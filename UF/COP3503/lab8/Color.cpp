#include "Color.h"

std::array<std::string, 16> Color::hexVals = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"};

Color::Color(int value, std::string name) :m_value(value), m_name(name) {}

void Color::SetValue(int value) {
    m_value = value;
}

void Color::SetName(const char* name) {
    m_name = name;
}

// Accessors 
unsigned char Color::GetR() const {
    int temp = m_value >> 16;
    return temp & 255;
}

unsigned char Color::GetG() const {
    int temp = m_value >> 8;
    return temp & 255;
}

unsigned char Color::GetB() const {
    return m_value & 255;
}

std::string Color::GetHexValue() const {
    return "0x" + hexValHelper(GetR()) + hexValHelper(GetG()) + hexValHelper(GetB());
}

std::string Color::hexValHelper(unsigned char val) const {
    return hexVals[val / 16] + hexVals[val % 16];
}
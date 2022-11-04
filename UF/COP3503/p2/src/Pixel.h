#pragma once
#include <fstream>

class Pixel {
    unsigned char blue;
    unsigned char green;
    unsigned char red;

    public:
    Pixel(std::ifstream& file);
    bool operator==(const Pixel& rhs);
    Pixel& operator*=(const Pixel& rhs);
    Pixel& operator-=(const Pixel& rhs);
    unsigned char clamp(int val);
};
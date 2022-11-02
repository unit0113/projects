#pragma once
#include <fstream>

class Pixel {
    unsigned char blue;
    unsigned char green;
    unsigned char red;

    public:
    Pixel(std::ifstream& file);
};
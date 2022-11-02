#pragma once
#include "Header.h"
#include "Pixel.h"
#include <fstream>
#include <vector>

class Image {
    Header header;
    std::vector<std::vector<Pixel>> pixels;

    public:
        Image(std::ifstream& file);
        short width() const {return header.width;};
        short height() const {return header.height;};
};
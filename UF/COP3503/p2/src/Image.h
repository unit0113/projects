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
        bool operator==(const Image& rhs);
        Image multiply(Image source, const Image& mask);
        Image subtract(Image source, const Image& mask);
        Image screen(Image source, const Image& mask);
        Image overlay(Image source, const Image& mask);
};
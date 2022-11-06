#pragma once
#include "Header.h"
#include "Pixel.h"
#include <fstream>
#include <vector>

class Image {
    Header header;
    std::vector<Pixel> pixels;

    public:
        Image(std::ifstream& file);
        short width() const {return header.width;};
        short height() const {return header.height;};
        bool operator==(const Image& rhs);
        bool operator!=(const Image& rhs);
        void multiply(const Image& mask);
        void subtract(const Image& mask);
        void screen(const Image& mask);
        void overlay(const Image& mask);
        void write(std::ofstream& file);
        Header getHeader() const {return header;};
        std::vector<Pixel> getPixels() const {return pixels;};
};
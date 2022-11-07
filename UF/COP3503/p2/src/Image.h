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
        void add(const Pixel& pixel);
        void screen(const Image& mask);
        void overlay(const Image& mask);
        void write(const char* filePath) const;
        void scale(const short blueScale, const short greenScale, const short redScale);

        Header getHeader() const {return header;};
        std::vector<Pixel> getPixels() const {return pixels;};
};

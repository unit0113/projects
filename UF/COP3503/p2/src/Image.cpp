#include "Image.h"

Image::Image(std::ifstream& file) : header(file) {
    size_t numPixels = header.width * header.height;
    for (size_t i{}; i < numPixels; ++i) {
        pixels.push_back(Pixel(file));
    }
}

bool Image::operator==(const Image& rhs) {
    if (header != rhs.header) {return false;}

    for (size_t i{}; i < pixels.size(); ++i) {
        if (pixels[i] != rhs.pixels[i]) {
            return false;
        }
    }
    return true;
}

bool Image::operator!=(const Image& rhs) {
    return !(*this == rhs);
}

void Image::multiply(const Image& mask) {
    for (size_t i{}; i < pixels.size(); ++i) {
        pixels[i] *= mask.pixels[i];
    }
}


void Image::subtract(const Image& mask) {
    for (size_t i{}; i < pixels.size(); ++i) {
        pixels[i] -= mask.pixels[i];
    }
}


void Image::screen(const Image& mask) {
    for (size_t i{}; i < pixels.size(); ++i) {
        pixels[i].screen(mask.pixels[i]);
    }
}


void Image::overlay(const Image& mask) {
    for (size_t i{}; i < pixels.size(); ++i) {
        pixels[i].overlay(mask.pixels[i]);
    }
}

void Image::write(const char* filePath) {
    std::ofstream file(filePath, std::ios::binary);
    header.write(file);
    for (Pixel p: pixels) {
        p.write(file);
    }
}
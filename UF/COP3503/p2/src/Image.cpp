#include "Image.h"

Image::Image(std::ifstream& file) {
    header = Header(file);

    for (short i{}; i < height(); ++i) {
        for (short j{}; j < width(); ++j) {
            pixels[i].push_back(Pixel(file));
        }
    }
}

bool Image::operator==(const Image& rhs) {
    if (header != rhs.header) {return false;}

    for (size_t i{}; i < pixels.size(), ++i) {
        for (size_t j{}; j < width(); ++j) {
            if (pixels[i][j] != rhs.pixels[i][j]) {
                return false;
            }
        }
    }
    return true;
}

Image Image::multiply(Image source, const Image& mask) {

}


Image Image::subtract(Image source, const Image& mask) {

}


Image Image::screen(Image source, const Image& mask) {

}


Image Image::overlay(Image source, const Image& mask) {

}
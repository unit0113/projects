#include "Image.h"

Image::Image(std::ifstream& file) : header(file) {
    while (file) {
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

Image Image::multiply(Image source, const Image& mask) {

}


Image Image::subtract(Image source, const Image& mask) {

}


Image Image::screen(Image source, const Image& mask) {

}


Image Image::overlay(Image source, const Image& mask) {

}
#include "Image.h"

Image::Image(std::ifstream& file) {
    header = Header(file);

    for (short i{}; i < height(); ++i) {
        for (short j{}; j < width(); ++j) {
            pixels[i].push_back(Pixel(file));
        }
    }
}
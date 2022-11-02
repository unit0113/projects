#include "Pixel.h"

Pixel::Pixel(std::ifstream& file) {
    blue = file.get();
    green = file.get();
    red = file.get();
}
#include "Pixel.h"

Pixel::Pixel(std::ifstream& file) {
    blue = file.get();
    green = file.get();
    red = file.get();
}

bool Pixel::operator==(const Pixel& rhs) {
    return blue == rhs.blue && green == rhs.green && red == rhs.red;
}

Pixel::Pixel& operator*=(const Pixel& rhs) {
    blue = (blue * rhs.red) / 255
    green = (green * rhs.red) / 255
    red = (red * rhs.red) / 255

    return *this;
}

Pixel::Pixel& operator-=(const Pixel& rhs) {
    int temp = blue - rhs.blue;
    blue = clamp(temp);
    temp = green - rhs.green;
    green = clamp(temp)
    temp = red - rhs.red;
    red = clamp(temp);
    
    return *this;
}

unsigned char Pixel::clamp(int val) {
    return std::max(std::min(255, val), 0)
}
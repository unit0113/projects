#include "Pixel.h"

Pixel::Pixel(std::ifstream& file) {
    m_blue = file.get();
    m_green = file.get();
    m_red = file.get();
}

//Test Constructor
Pixel::Pixel(unsigned char blue, unsigned char green, unsigned char red)
    : m_blue(blue), m_green(green), m_red(red) {}

bool Pixel::operator==(const Pixel& rhs) {
    return m_blue == rhs.m_blue && m_green == rhs.m_green && m_red == rhs.m_red;
}

Pixel& Pixel::operator*=(const Pixel& rhs) {
    m_blue = (m_blue * rhs.m_red) / 255;
    m_green = (m_green * rhs.m_red) / 255;
    m_red = (m_red * rhs.m_red) / 255;

    return *this;
}

Pixel& Pixel::operator-=(const Pixel& rhs) {
    int temp = m_blue - rhs.m_blue;
    m_blue = clamp(temp);
    temp = m_green - rhs.m_green;
    m_green = clamp(temp);
    temp = m_red - rhs.m_red;
    m_red = clamp(temp);
    
    return *this;
}

unsigned char Pixel::clamp(int val) {
    return std::max(std::min(255, val), 0);
}
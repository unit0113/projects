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
    m_blue = 0.5f + (m_blue * rhs.m_blue) / 255;
    m_green = 0.5f + (m_green * rhs.m_green) / 255;
    m_red = 0.5f + (m_red * rhs.m_red) / 255;

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

void Pixel::screen(const Pixel& p) {
    m_blue = screenHelper(m_blue, p.m_blue);
    m_green = screenHelper(m_green, p.m_green);
    m_red = screenHelper(m_red, p.m_red);
}

unsigned char Pixel::screenHelper(const unsigned char& val1, const unsigned char& val2) {
    float temp = 1 - (1 - val1 / 255.0) * (1 - val2 / 255.0);
    return 0.5f + 255 * temp;
}

void Pixel::overlay(const Pixel& p) {
    if (p.isDark()) {
        m_blue = overlayHelperMultiply(m_blue, p.m_blue);
        m_green = overlayHelperMultiply(m_green, p.m_green);
        m_red = overlayHelperMultiply(m_red, p.m_red);
    } else {
        m_blue = overlayHelperScreen(m_blue, p.m_blue);
        m_green = overlayHelperScreen(m_green, p.m_green);
        m_red = overlayHelperScreen(m_red, p.m_red);
    }
}

unsigned char Pixel::overlayHelperMultiply(const unsigned char& val1, const unsigned char& val2) {
    int temp = 0.5f + 2 * (val1 * val2) / 255;
    return clamp(temp);
}

unsigned char Pixel::overlayHelperScreen(const unsigned char& val1, const unsigned char& val2) {
    float temp = 1 - 2* (1 - val1 / 255.0) * (1 - val2 / 255.0);
    return 0.5f + 255 * temp;
}

bool Pixel::isDark() const {
    float temp = m_blue / 255.0f + m_green / 255.0f + m_red / 255.0f;
    return ((temp / 3.0) <= 0.5);
}
#include "Pixel.h"
#include <algorithm>

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

bool Pixel::operator!=(const Pixel& rhs) {
    return !(*this == rhs);
}

Pixel& Pixel::operator*=(const Pixel& rhs) {
    m_blue = 0.5f + m_blue * rhs.m_blue / 255.0f;
    m_green = 0.5f + m_green * rhs.m_green / 255.0f;
    m_red = 0.5f + m_red * rhs.m_red / 255.0f;

    return *this;
}

Pixel& Pixel::operator-=(const Pixel& rhs) {
    m_blue = clamp((int)m_blue - (int)rhs.m_blue);
    m_green = clamp((int)m_green - (int)rhs.m_green);
    m_red = clamp((int)m_red - (int)rhs.m_red);
    
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
    float temp = 1 - (1 - val1 / 255.0f) * (1 - val2 / 255.0f);
    return 0.5f + 255 * temp;
}

void Pixel::overlay(const Pixel& mask) {
    if (isDark(mask.m_blue)) {
        m_blue = overlayHelperMultiply(m_blue, mask.m_blue);
    } else {
        m_blue = overlayHelperScreen(m_blue, mask.m_blue);
    }

    if (isDark(mask.m_green)) {
        m_green = overlayHelperMultiply(m_green, mask.m_green);
    } else {
        m_green = overlayHelperScreen(m_green, mask.m_green);
    }

    if (isDark(mask.m_red)) {
        m_red = overlayHelperMultiply(m_red, mask.m_red);
    } else {
        m_red = overlayHelperScreen(m_red, mask.m_red);
    }
}

unsigned char Pixel::overlayHelperMultiply(const unsigned char& val1, const unsigned char& val2) {
    int temp = 0.5f + 2 * (val1 * val2) / 255.0f;
    return clamp(temp);
}

unsigned char Pixel::overlayHelperScreen(const unsigned char& val1, const unsigned char& val2) {
    float temp = 1 - 2 * (1 - val1 / 255.0f) * (1 - val2 / 255.0f);
    return 0.5f + 255.0f * temp;
}

bool Pixel::isDark(const unsigned char& val) const {
    return (val / 255.0f) <= 0.5f;
}

void Pixel::write(std::ofstream& file) {
    file.write(reinterpret_cast<const char*>(&m_blue), sizeof(m_blue));
    file.write(reinterpret_cast<const char*>(&m_green), sizeof(m_green));
    file.write(reinterpret_cast<const char*>(&m_red), sizeof(m_red));
}
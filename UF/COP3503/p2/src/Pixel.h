#pragma once
#include <fstream>

class Pixel {
    unsigned char m_blue;
    unsigned char m_green;
    unsigned char m_red;

    public:
    Pixel(std::ifstream& file);
    Pixel(unsigned char blue, unsigned char green, unsigned char red);
    bool operator==(const Pixel& rhs);
    Pixel& operator*=(const Pixel& rhs);
    Pixel& operator-=(const Pixel& rhs);
    unsigned char clamp(int val);
    void screen(const Pixel& p);
    unsigned char screenHelper(const unsigned char& val1, const unsigned char& val2);
    void overlay(const Pixel& p);
    unsigned char overlayHelperMultiply(const unsigned char& val1, const unsigned char& val2);
    unsigned char overlayHelperScreen(const unsigned char& val1, const unsigned char& val2);
    bool isDark() const;

    unsigned char getBlue() const {return m_blue;};
    unsigned char getGreen() const {return m_green;};
    unsigned char getRed() const {return m_red;};
};
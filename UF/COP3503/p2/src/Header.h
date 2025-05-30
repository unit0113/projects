#pragma once
#include <fstream>

struct Header {
    char idLength;
    char colorMapType;
    char dataTypeCode;
    short colorMapOrigin;
    short colorMapLength;
    char colorMapDepth;
    short xOrigin;
    short yOrigin;
    short width;
    short height;
    char bitsPerPixel;
    char imageDescriptor;

    Header(std::ifstream& file);
    Header(const Header& other) = default;
    bool operator==(const Header& rhs);
    bool operator!=(const Header& rhs);
    void write(std::ofstream& file) const;
};

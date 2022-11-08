#include <algorithm>
#include "Image.h"

Image::Image(std::ifstream& file) : header(file) {
    const size_t numPixels = header.width * header.height;
    for (size_t i{}; i < numPixels; ++i) {
        pixels.push_back(Pixel(file));
    }
}


Image::Image(const Image& blue, const Image& green, const Image& red) : header(blue.getHeader()) {
    std::vector<Pixel> bluePixels = blue.getPixels();
    std::vector<Pixel> greenPixels = green.getPixels();
    std::vector<Pixel> redPixels = red.getPixels();

    const size_t numPixels = bluePixels.size();
    for (size_t i{}; i < numPixels; ++i) {
        pixels.push_back(Pixel(bluePixels[i].getBlue(), greenPixels[i].getGreen(), redPixels[i].getRed()));
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


void Image::multiply(const Image& mask) {
    for (size_t i{}; i < pixels.size(); ++i) {
        pixels[i] *= mask.pixels[i];
    }
}


void Image::subtract(const Image& mask) {
    for (size_t i{}; i < pixels.size(); ++i) {
        pixels[i] -= mask.pixels[i];
    }
}


void Image::add(const Pixel& inPixel) {
    for (Pixel& p: pixels) {
        p += inPixel;
    }
}


void Image::screen(const Image& mask) {
    for (size_t i{}; i < pixels.size(); ++i) {
        pixels[i].screen(mask.pixels[i]);
    }
}


void Image::overlay(const Image& mask) {
    for (size_t i{}; i < pixels.size(); ++i) {
        pixels[i].overlay(mask.pixels[i]);
    }
}


void Image::write(const char* filePath) const {
    std::ofstream file(filePath, std::ios::binary);
    header.write(file);
    for (const Pixel& p: pixels) {
        p.write(file);
    }
}


void Image::writeChannel(const char* blueFilePath, const char* greenFilePath, const char* redFilePath) const {
    std::ofstream file(blueFilePath, std::ios::binary);
    header.write(file);
    for (const Pixel& p: pixels) {
        p.writeBlue(file);
    }
    file.close();

    file.open(greenFilePath, std::ios::binary);
    header.write(file);
    for (const Pixel& p: pixels) {
        p.writeGreen(file);
    }
    file.close();

    file.open(redFilePath, std::ios::binary);
    header.write(file);
    for (const Pixel& p: pixels) {
        p.writeRed(file);
    }
}


void Image::scale(const short blueScale, const short greenScale, const short redScale) {
    for (Pixel& p: pixels) {
        p.scale(blueScale, greenScale, redScale);
    }
}


void Image::flip() {
    std::reverse(pixels.begin(), pixels.end());
}
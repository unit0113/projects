#include "Header.h"

Header::Header(std::ifstream& file) {
        file.read(&idLength, sizeof(idLength));
        file.read(&colorMapType, sizeof(colorMapType));
        file.read(&dataTypeCode, sizeof(dataTypeCode));
        file.read(reinterpret_cast<char*>(&colorMapOrigin), sizeof(colorMapOrigin));
        file.read(reinterpret_cast<char*>(&colorMapLength), sizeof(colorMapLength));
        file.read(&colorMapDepth, sizeof(colorMapDepth));
        file.read(reinterpret_cast<char*>(&xOrigin), sizeof(xOrigin));
        file.read(reinterpret_cast<char*>(&yOrigin), sizeof(yOrigin));
        file.read(reinterpret_cast<char*>(&width), sizeof(width));
        file.read(reinterpret_cast<char*>(&height), sizeof(height));
        file.read(&bitsPerPixel, sizeof(bitsPerPixel));
        file.read(&imageDescriptor, sizeof(imageDescriptor));
        file.read(&imageDescriptor, sizeof(imageDescriptor));
    }
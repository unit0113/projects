#include <iostream>
#include <iomanip>
#include "Image.h"
using namespace std;

static int testCaseName = 30;

void testPixelMult();
void testPixelSubtract();
void testPixelScreen();
void testPixelOverlayDark();
void testPixelOverlayLight();
void testHeaderEqual();

int main() {
    testPixelMult();
    testPixelSubtract();
    testPixelScreen();
    testPixelOverlayDark();
    testPixelOverlayLight();
    testHeaderEqual();
}



void testPixelMult() {
    Pixel p1 = Pixel(100, 100, 100);
    Pixel p2 = Pixel(100, 0, 255);
    p1 *= p2;

    cout << setfill('.') << setw(testCaseName) << left << "Test Pixel Multiply";
    Pixel p3 = Pixel(39, 0, 100);
    if (p1 == p3) {
        cout << "Passed\n";
    } else {
        cout << "Failed\n";
        cout << "\tPixel 1: [" << p1.getBlue() << ' ' << p1.getGreen() << ' ' << p1.getRed() << "]\n";
        cout << "\tPixel 2: [" << p3.getBlue() << ' ' << p3.getGreen() << ' ' << p3.getRed() << "]\n";
    }
}

void testPixelSubtract() {
    Pixel p1 = Pixel(100, 100, 100);
    Pixel p2 = Pixel(100, 0, 255);
    p1 -= p2;

    cout << setfill('.') << setw(testCaseName) << left << "Test Pixel Subtract";
    Pixel p3 = Pixel(0, 100, 0);
    if (p1 == p3) {
        cout << "Passed\n";
    } else {
        cout << "Failed\n";
        cout << "\tPixel 1: [" << p1.getBlue() << ' ' << p1.getGreen() << ' ' << p1.getRed() << "]\n";
        cout << "\tPixel 2: [" << p3.getBlue() << ' ' << p3.getGreen() << ' ' << p3.getRed() << "]\n";
    }
}

void testPixelScreen() {
    Pixel p1 = Pixel(100, 100, 100);
    Pixel p2 = Pixel(100, 0, 255);
    p1.screen(p2);

    cout << setfill('.') << setw(testCaseName) << left << "Test Pixel Screen";
    Pixel p3 = Pixel(161, 100, 255);
    if (p1 == p3) {
        cout << "Passed\n";
    } else {
        cout << "Failed\n";
        cout << "\tPixel 1: [" << (int)p1.getBlue() << ' ' << (int)p1.getGreen() << ' ' << (int)p1.getRed() << "]\n";
        cout << "\tPixel 2: [" << (int)p3.getBlue() << ' ' << (int)p3.getGreen() << ' ' << (int)p3.getRed() << "]\n";
    }
}

void testPixelOverlayDark() {
    Pixel p1 = Pixel(100, 100, 100);
    Pixel p2 = Pixel(0, 50, 100);
    p1.overlay(p2);

    cout << setfill('.') << setw(testCaseName) << left << "Test Pixel Overlay-Dark";
    Pixel p3 = Pixel(0, 39, 78);
    if (p1 == p3) {
        cout << "Passed\n";
    } else {
        cout << "Failed\n";
        cout << "\tPixel 1: [" << (int)p1.getBlue() << ' ' << (int)p1.getGreen() << ' ' << (int)p1.getRed() << "]\n";
        cout << "\tPixel 2: [" << (int)p3.getBlue() << ' ' << (int)p3.getGreen() << ' ' << (int)p3.getRed() << "]\n";
    }
}

void testPixelOverlayLight() {
    Pixel p1 = Pixel(100, 100, 100);
    Pixel p2 = Pixel(100, 175, 255);
    p1.overlay(p2);

    cout << setfill('.') << setw(testCaseName) << left << "Test Pixel Overlay-Light";
    Pixel p3 = Pixel(67, 158, 255);
    if (p1 == p3) {
        cout << "Passed\n";
    } else {
        cout << "Failed\n";
        cout << "\tPixel 1: [" << (int)p1.getBlue() << ' ' << (int)p1.getGreen() << ' ' << (int)p1.getRed() << "]\n";
        cout << "\tPixel 2: [" << (int)p3.getBlue() << ' ' << (int)p3.getGreen() << ' ' << (int)p3.getRed() << "]\n";
    }
}

void testHeaderEqual() {
    const int varLength = 20;

    ifstream file("input/cat.tga", ios::binary);
    Header h1(file);
    file.close();

    file.open("input/cat.tga");
    Header h2(file);

    cout << setfill('.') << setw(testCaseName) << left << "Test Header Equals";
    if (h1 == h2) {
        cout << "Passed\n";
    } else {
        cout << "Failed\n";
        cout << "\t" << setw(varLength) << "IDLength:" << (int)h1.idLength << " / " << (int)h2.idLength << endl;
        cout << "\t" << setw(varLength) << "Color Map Type:" << (int)h1.colorMapType << " / " << (int)h2.colorMapType << endl;
        cout << "\t" << setw(varLength) << "Data Type Code:" << (int)h1.dataTypeCode << " / " << (int)h2.dataTypeCode << endl;
        cout << "\t" << setw(varLength) << "Color Map Origin:" << h1.colorMapOrigin << " / " << h2.colorMapOrigin << endl;
        cout << "\t" << setw(varLength) << "Color Map Length:" << h1.colorMapLength << " / " << h2.colorMapLength << endl;
        cout << "\t" << setw(varLength) << "Color Map Depth:" << (int)h1.colorMapDepth << " / " << (int)h2.colorMapDepth << endl;
        cout << "\t" << setw(varLength) << "X Origin:" << h1.xOrigin << " / " << h2.xOrigin << endl;
        cout << "\t" << setw(varLength) << "Y Origin:" << h1.yOrigin << " / " << h2.yOrigin << endl;
        cout << "\t" << setw(varLength) << "Width:" << h1.width << " / " << h2.width << endl;
        cout << "\t" << setw(varLength) << "Height:" << h1.height << " / " << h2.height << endl;
        cout << "\t" << setw(varLength) << "Bits Per Pixel:" << (int)h1.bitsPerPixel << " / " << (int)h2.bitsPerPixel << endl;
        cout << "\t" << setw(varLength) << "Image Descriptor:" << (int)h1.imageDescriptor << " / " << (int)h2.imageDescriptor << endl;
    }
}
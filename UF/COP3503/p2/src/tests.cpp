#include <iostream>
#include <iomanip>
#include "Image.h"
using namespace std;

void testPixelMult();

int main() {
    testPixelMult();
}



void testPixelMult() {
    Pixel p1 = Pixel(100, 100, 100);
    Pixel p2 = Pixel(100, 0, 255);
    p1 *= p2;

    cout << setfill('.') << setw(25) << left << "Test Pixel Multiply";
    Pixel p3 = Pixel(39, 0, 100);
    if (p1 == p3) {
        cout << "Passed\n";
    } else {
        cout << "Failed\n";
        cout << p1.getBlue() << ' ' << p1.getGreen() << ' ' << p1.getRed() << "//" << p3.getBlue() << ' ' << p3.getGreen() << ' ' << p3.getRed() << endl;
    }
}

#include "Image.h"
#include <fstream>
#include <stdexcept>
using namespace std;

void part1();

int main() {
    part1();
}


void part1() {
    ifstream file("input/layer1.tga", ios::binary);
    if (!file) {
        throw runtime_error("File not found");
    }
    Image source(file);
    file.close();

    file.open("input/pattern1.tga", ios::binary);
    if (!file) {
        throw runtime_error("File not found");
    }
    Image mask(file);
    file.close();

    source.multiply(mask);
    ofstream outFile("output/part1.tga", ios::binary);
    source.write(outFile);
}
#include "Image.h"
#include <fstream>
#include <stdexcept>
using namespace std;

Image openFile(const char* file);
void part1();
void part2();
void part3();
void part4();
void part5();

int main() {
    part1();
    part2();
    part3();
    part4();
    part5();
}


Image openFile(const char* filePath) {
    ifstream file(filePath, ios::binary);
    if (!file) {
        throw runtime_error("File not found");
    }
    Image img(file);
    file.close();
    return img;
}


void part1() {
    Image source = openFile("input/layer1.tga");
    Image mask = openFile("input/pattern1.tga");

    source.multiply(mask);
    source.write("output/part1.tga");
}


void part2() {
    Image top = openFile("input/layer2.tga");
    Image bottom = openFile("input/car.tga");

    bottom.subtract(top);
    bottom.write("output/part2.tga");
}


void part3() {
    Image base = openFile("input/layer1.tga");
    Image mask = openFile("input/pattern2.tga");
    Image screen = openFile("input/text.tga");

    base.multiply(mask);
    base.screen(screen);
    base.write("output/part3.tga");
}


void part4() {
    Image base = openFile("input/layer2.tga");
    Image mask = openFile("input/circles.tga");
    Image sub = openFile("input/pattern2.tga");

    base.multiply(mask);
    base.subtract(sub);
    base.write("output/part4.tga");
}


void part5() {
    Image base = openFile("input/layer1.tga");
    Image mask = openFile("input/pattern1.tga");

    base.overlay(mask);
    base.write("output/part5.tga");
}
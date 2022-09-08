#include <iostream>
#include <string>
#include <sstream>
using namespace std;


class Box {
public:
    double length, width, height;
    string name;
    Box() {
        length = 1;
        width = 1;
        height = 1;
    }
    Box(double l, double w, double h) {
        length = l;
        width = w;
        height = h;
    }

    Box& operator ++() {
        length++;
        width++;
        height++;
        return *this;
    }

    Box& operator ++(int) {
        length++;
        width++;
        height++;
        return *this;
    }

    Box& operator --() {
        length--;
        width--;
        height--;
        return *this;
    }

    Box& operator --(int) {
        length--;
        width--;
        height--;
        return *this;
    }

    operator const char*() {
        ostringstream boxStream;
        boxStream << "Box: " << length << ", " << width << ", " << height;
        name = boxStream.str();
        return name.c_str();
    }

    Box operator + (const Box& b2) {
        Box newBox;
        newBox.length = length + b2.length;
        newBox.width = width + b2.width;
        newBox.height = height + b2.height;
        return newBox;
    }

    double operator [] (int x) {
        if (x == 0) {
            return length;
        } else if (x == 1) {
            return width;
        } else if (x == 2) {
            return height;
        } else {
            return 0;
        }
    }

    bool operator == (const Box& b2) {
        return ((length == b2.length)
                && (width == b2.width)
                && (height == b2.height));
    }

    bool operator < (const Box& b2) {
        return (length + width + height) < (b2.length + b2.width + b2.height);
    }

    bool operator > (const Box& b2) {
        return (length + width + height) > (b2.length + b2.width + b2.height);
    }

    void operator = (const Box& b2) {
        length = b2.length;
        width = b2. width;
        height = b2.height;
    }
};

int main() {
    Box box(10, 10, 10);
    cout << box << endl;
    box++;
    cout << box << endl;
    Box box2(5, 5, 5);
    cout << box + box2 << endl;
    cout << "Length: " << box[0] << endl;
    cout << boolalpha;
    cout << "Boxes are equal: " << (box == box2) << endl;
    cout << (box2 < box) << endl;
    box = box2;
    cout << "Boxes are equal: " << (box == box2) << endl;









}
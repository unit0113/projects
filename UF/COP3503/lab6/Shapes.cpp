#include "Shapes.h"
#include <iostream>
#include <iomanip>
#include <compare>
#include <math.h>
#include <string>
using namespace std;


void Shape2D::ShowArea() const {
    cout << "The area of the " << GetName2D() << " is : " << Area() << endl;
}

strong_ordering Shape2D::operator<=>(const Shape2D &rhs) const {
    if (Area() < rhs.Area()) {
        return strong_ordering::less;
    } else if (Area() > rhs.Area()) {
        return strong_ordering::greater;
    } else {
        return strong_ordering::equal;
    }
}

/*---------------Square-----------------*/
float Square::Area() const {
    return pow(m_side, 2);
}

string Square::GetName2D() const {
    return "Square";
}

void Square::Scale(float scaleFactor) {
    m_side *= scaleFactor;
}

void Square::Display() const {
    cout << fixed << setprecision(2);
    Shape2D::ShowArea();
    cout << "Length of a side: " << m_side << endl;
}

/*---------------Triangle-----------------*/
float Triangle::Area() const {
    return m_base * m_height / 2.0f;
}

string Triangle::GetName2D() const {
    return "Triangle";
}

void Triangle::Scale(float scaleFactor) {
    m_base *= scaleFactor;
    m_height *= scaleFactor;
}

void Triangle::Display() const {
    cout << fixed << setprecision(2);
    Shape2D::ShowArea();
    cout << "Base: " << m_base << endl;
    cout << "Height: " << m_height << endl;
}

/*---------------Circle-----------------*/
float Circle::Area() const {
    return PI * pow(m_radius, 2);
}

string Circle::GetName2D() const {
    return "Circle";
}

void Circle::Scale(float scaleFactor) {
    m_radius *= scaleFactor;
}

void Circle::Display() const {
    cout << fixed << setprecision(2);
    Shape2D::ShowArea();
    cout << "Radius: " << m_radius << endl;
}
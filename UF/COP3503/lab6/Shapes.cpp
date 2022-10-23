#include "Shapes.h"
#include <iostream>
#include <iomanip>
#include <math.h>
#include <string>
using namespace std;

/*---------------2D-----------------*/
void Shape2D::ShowArea() const {
    cout << "The area of the " << GetName2D() << " is : " << Area() << endl;
}

bool Shape2D::operator>(const Shape2D &rhs) const {
    return Area() > rhs.Area();
}

bool Shape2D::operator<(const Shape2D &rhs) const {
    return Area() < rhs.Area();
}

bool Shape2D::operator==(const Shape2D &rhs) const {
    return Area() == rhs.Area();
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

/*---------------3D-----------------*/
void Shape3D::ShowVolume() const {
    cout << "The volume of the " << GetName3D() << " is : " << Volume();
}

bool Shape3D::operator>(const Shape3D &rhs) const {
    return Volume() > rhs.Volume();
}

bool Shape3D::operator<(const Shape3D &rhs) const {
    return Volume() < rhs.Volume();
}

bool Shape3D::operator==(const Shape3D &rhs) const {
    return Volume() == rhs.Volume();
}


/*---------------Pyramid-----------------*/
float TriangularPyramid::Volume() const {
    return m_tri.Area() * m_height / 3.0f;
}

string TriangularPyramid::GetName3D() const {
    return "TriangularPyramid";
}

void TriangularPyramid::Scale(float scaleFactor) {
    m_height *= scaleFactor;
    m_tri.Scale(scaleFactor);
}

void TriangularPyramid::Display() const {
    cout << fixed << setprecision(2);
    Shape3D::ShowVolume();
    cout << "The height is: " << m_height << endl;
    m_tri.Display();
}


/*---------------Cylinder-----------------*/
float Cylinder::Volume() const {
    return m_cir.Area() * m_height;
}

string Cylinder::GetName3D() const {
    return "Cylinder";
}

void Cylinder::Scale(float scaleFactor) {
    m_height *= scaleFactor;
    m_cir.Scale(scaleFactor);
}

void Cylinder::Display() const {
    cout << fixed << setprecision(2);
    Shape3D::ShowVolume();
    cout << "The height is: " << m_height << endl;
    m_cir.Display();
}


/*---------------Sphere-----------------*/
float Sphere::Volume() const {
    return m_cir.Area() * m_radius * 4.0f / 3.0f;
}

string Sphere::GetName3D() const {
    return "Sphere";
}

void Sphere::Scale(float scaleFactor) {
    m_radius *= scaleFactor;
    m_cir.Scale(scaleFactor);
}

void Sphere::Display() const {
    cout << fixed << setprecision(2);
    Shape3D::ShowVolume();
    m_cir.Display();
}
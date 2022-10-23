#pragma once
#include <string>

const float PI = 3.14159f;

class Shape {
    public:
        virtual ~Shape() = 0;
        virtual void Scale(float scaleFactor) = 0;
        virtual void Display() const = 0;
};


class Shape2D: virtual public Shape { 
    public:
        virtual ~Shape2D() = 0; 
        virtual float Area() const = 0; 
        void ShowArea() const; 
        virtual std::string GetName2D() const = 0;

        bool operator>(const Shape2D &rhs) const; 
        bool operator<(const Shape2D &rhs) const; 
        bool operator==(const Shape2D &rhs) const; 
};


class Square: virtual public Shape2D {
    float m_side;

    public:
        Square(float side = 0.0f) : m_side(side){};
        virtual ~Square() = default; 
        virtual float Area() const override;
        virtual std::string GetName2D() const override;
        virtual void Scale(float scaleFactor) override;
        virtual void Display() const override;
};


class Triangle: virtual public Shape2D {
    float m_base;
    float m_height;

    public:
        Triangle(float base = 0.0f, float height = 0.0f) : m_base(base), m_height(height){};
        virtual ~Triangle() = default; 
        virtual float Area() const override;
        virtual std::string GetName2D() const override;
        virtual void Scale(float scaleFactor) override;
        virtual void Display() const override;
};


class Circle: virtual public Shape2D {
    float m_radius;

    public:
        Circle(float radius = 0.0f) : m_radius(radius){};
        virtual ~Circle() = default; 
        virtual float Area() const override;
        virtual std::string GetName2D() const override;
        virtual void Scale(float scaleFactor) override;
        virtual void Display() const override;
};



class Shape3D: virtual public Shape { 
    public :
        virtual ~Shape3D() = 0; 
        virtual float Volume() const = 0; 
        void ShowVolume() const; 
        virtual std::string GetName3D() const = 0;

        bool operator>(const Shape3D &rhs) const; 
        bool operator<(const Shape3D &rhs) const; 
        bool operator==(const Shape3D &rhs) const;
};

class TriangularPyramid: virtual public Shape3D, private Triangle {
    Triangle m_tri;
    float m_height;

    public:
        TriangularPyramid(float height = 0.0f, float base = 0.0f, float tri_height = 0.0f) : m_height(height), m_tri(base, tri_height){};
        virtual ~TriangularPyramid() = default; 
        virtual float Volume() const override;
        virtual std::string GetName3D() const override;
        virtual void Scale(float scaleFactor) override;
        virtual void Display() const override;
};


class Cylinder: virtual public Shape3D, private Circle {
    Circle m_cir;
    float m_height;

    public:
        Cylinder(float height = 0.0f, float radius = 0.0f) : m_height(height), m_cir(radius){};
        virtual ~Cylinder() = default; 
        virtual float Volume() const override;
        virtual std::string GetName3D() const override;
        virtual void Scale(float scaleFactor) override;
        virtual void Display() const override;
};


class Sphere: virtual public Shape3D, private Circle {
    Circle m_cir;
    float m_radius;

    public:
        Sphere(float radius = 0.0f) : m_radius(radius), m_cir(radius){};
        virtual ~Sphere() = default; 
        virtual float Volume() const override;
        virtual std::string GetName3D() const override;
        virtual void Scale(float scaleFactor) override;
        virtual void Display() const override;
};
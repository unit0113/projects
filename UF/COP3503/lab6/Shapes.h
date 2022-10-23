#pragma once
#include <compare>

const float PI = 3.14159f;

class Shape {
    public:
        virtual ~Shape() = default;
        virtual void Scale(float scaleFactor) = 0;
        virtual void Display() const = 0;
};


class Shape2D: virtual public Shape { 
    public:
        virtual ~Shape2D() = default; 
        virtual float Area() const = 0; 
        void ShowArea() const; 
        virtual string GetName2D() const = 0;

        strong_ordering operator<=>(const Shape2D &rhs) const; 
};


class Square: virtual public Shape2D {
    float m_side;

    public:
        Square(float side) : m_side(side){};
        virtual ~Square() = default; 
        virtual float Area() const override;
        virtual string GetName2D() const override;
        virtual void Scale(float scaleFactor) override;
        virtual void Display() const override;
};


class Triangle: virtual public Shape2D {
    float m_base;
    float m_height;

    public:
        Triangle(float base, float height) : m_base(base), m_height(height){};
        virtual ~Triangle() = default; 
        virtual float Area() const override;
        virtual string GetName2D() const override;
        virtual void Scale(float scaleFactor) override;
        virtual void Display() const override;
};


class Circle: virtual public Shape2D {
    float m_radius;

    public:
        Circle(float radius) : m_radius(radius){};
        virtual ~Circle() = default; 
        virtual float Area() const override;
        virtual string GetName2D() const override;
        virtual void Scale(float scaleFactor) override;
        virtual void Display() const override;
};



class Shape3D: virtual public Shape { 
    public :
        virtual ~Shape3D() = default; 
        virtual float Volume() const = 0; 
        void ShowVolume() const; 
        virtual string GetName3D() const = 0;

        strong_ordering operator<=>(const Shape3D &rhs) const; 
};


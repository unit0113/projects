#pragma once
#include <iostream>

class Shape {
    public:
        virtual ~Shape() {}
        virtual void Display() const = 0;
};

class testShape: public Shape {
    public:
        void Display() const override;
};

class ABC {
    public:
    virtual void DoSomething() = 0;
    void Foo() {std::cout << "Hello world\n";}
};

class CC: public ABC {
    public:
    void DoSomething() override;
};

class OC {
    public:
    void DoSomething();
};
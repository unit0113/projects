#include <iostream>

// used for plain old data structures (point)
struct mystruct {
    int i;
    std::string s;
    bool b;

    // can put methods, but shouldn't
    void test() {
        std::cout << "Test" << std::endl;
    }
};

struct person {
    std::string name;
    int age;
    char gender;

    void print_info() {
        std::cout << "Name: " << name << ", Age: " << age << ", Gender: " << gender << std::endl;
    }
};

int main() {

    struct mystruct bob;
    bob.i = 20;
    bob.s = "Hi";
    bob.b = true;

    std::cout << bob.i << std::endl;
    std::cout << bob.s << std::endl;
    std::cout << bob.b << std::endl;
    bob.test();

    struct person p1;
    p1.name = "Max";
    p1.age = 25;
    p1.gender = 'm';

    struct person p2;
    p2.name = "Anna";
    p2.age = 22;
    p2.gender = 'f';

    p1.print_info();
    p2.print_info();

    std::cout << sizeof(p1) << std::endl;

    return 0;
}
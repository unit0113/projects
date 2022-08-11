#include <iostream>

inline int product(int num1, int num2);

template<class T>
void swap(T &val1, T &val2);

template<class T>
T product2(T val1, T val2);


int main() {

    std::cout << product(5, 3) << std::endl;

    char c1 = 'a';
    char c2 = 'e';
    swap(c1, c2);
    std::cout << c1 << ' ' << c2 << std::endl;

    double d1 = 4.5;
    double d2 = 1.5;
    std::cout << product2(d1, d2) << std::endl;

    int i1 = 5;
    int i2 = 10;
    std::cout << product2(i1, i2) << std::endl;

}


inline int product(int num1, int num2) {
    return num1 * num2;
}


template<class T>
void swap(T &val1, T &val2) {
    T temp = val1;
    val1 = val2;
    val2 = temp;
}


template<class T>
T product2(T val1, T val2) {
    return val1 * val2;
}

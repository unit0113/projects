#include <iostream>

class Vector3f {
    public:
        Vector3f() {
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
        }

        Vector3f(float x_in, float y_in, float z_in) {
            x = x_in;
            y = y_in;
            z = z_in;
        }

        Vector3f operator+(const Vector3f &other_vec) const {
            Vector3f result_vec;
            result_vec.x = x + other_vec.x;
            result_vec.y = y + other_vec.y;
            result_vec.z = z + other_vec.z;

            return result_vec;
        }

        // Pre-increment
        Vector3f operator++() {
            x++;
            y++;
            z++;
            return *this;
        }

        // Post-increment
        Vector3f operator++(int) {
            Vector3f old = *this;
            operator++();
            return old;
        }

        bool operator==(const Vector3f &oth_vec) const {
            return (x == oth_vec.x) && (y == oth_vec.y) && (z == oth_vec.z);
        }

        float get_x() {
            return x;
        }

        float get_y() {
            return y;
        }

        float get_z() {
            return z;
        }

        void print_vec() {
            std::cout << "X: " << x << ", Y: " << y << ", Z: " << z << std::endl;
        }

        float x;
        float y;
        float z;
};

std::ostream& operator<<(std::ostream& os, const Vector3f& obj) {
    os << "X: " << obj.x << ", Y: " << obj.y << ", Z: " << obj.z;
    return os;
}

int main() {

    Vector3f vec1(1.5f, 2.5f, 3.5f);
    Vector3f vec2(4.0f, 5.0f, 6.0f);

    Vector3f res_vec = vec1 + vec2;

    res_vec.print_vec();

    res_vec++;
    res_vec.print_vec();

    std::cout << res_vec++ << std::endl;
    std::cout << res_vec << std::endl;

    std::cout << (res_vec == res_vec) << std::endl;
    std::cout << (res_vec == vec1) << std::endl;

    return 0;
}
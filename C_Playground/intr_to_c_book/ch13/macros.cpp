#include <iostream>


#define PI 3.14159
#define VOLUME_SPHERE(x) ((4/3.0) * PI * (x) * (x) * (x))
#define SUM(x, y) ((x) + (y))
#define MIN2(x, y) ((x) < (y) ? (x) : (y))
#define MIN3(x, y, z) MIN2(x, MIN2(y, z))
#define PRINT(s) std::cout<<(s)<<std::endl


int main() {

    std::cout << VOLUME_SPHERE(2) << std::endl;
    std::cout << SUM(7, 6) << std::endl;
    std::cout << MIN2(5, 20) << std::endl;
    std::cout << MIN3(13, 4, 53) << std::endl;
    PRINT("Hello, world");

    return 0;
}
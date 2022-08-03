#include <iostream>
#include <iomanip>


int main() {

    const int num_grades = 5;
    std::cout << "Enter Maths Grades: ";
    double math[num_grades] = {};
    double math_best = 0;
    double math_worst = 1000;
    double math_sum = 0;
    for (size_t i = 0; i < num_grades; i++) {
        std::cin >> math[i];
        math_best = std::max(math_best, math[i]);
        math_worst = std::min(math_worst, math[i]);
        math_sum += math[i];
    }

    std::cout << "Enter Physics Grades: ";
    double physics[num_grades] = {};
    double physics_best = 0;
    double physics_worst = 1000;
    double physics_sum = 0;
    for (size_t i = 0; i < num_grades; i++) {
        std::cin >> physics[i];
        physics_best = std::max(physics_best, physics[i]);
        physics_worst = std::min(physics_worst, physics[i]);
        physics_sum += physics[i];
    }


    auto width = std::setw(6);
    auto large_width = std::setw(8);
    auto title_width = std::setw(10);

    std::cout << "        ";
    for (size_t i = 1; i <= num_grades; i++) {
        std::cout << width << i;
    }
    std::cout << "     BEST    WORST  AVERAGE" << std::endl;
    std::cout << title_width << "" << std::string(56,'*') << '\n';
    std::cout << std::setprecision(2) << std::fixed;

    std::cout << title_width << std::left << "Math" << std::right;
    for (double grade: math) {
        std::cout << width << grade;
    }
    std::cout << large_width << math_best << large_width << math_worst << large_width << math_sum / num_grades << std::endl;

    std::cout << title_width << std::left << "Physics" << std::right;
    for (double grade: physics) {
        std::cout << width << grade;
    }
    std::cout << large_width << physics_best << large_width << physics_worst << large_width << physics_sum / num_grades << std::endl;    

    return 0;
}
#include <iostream>


int main() {

    int num_students = 0;
    std::cout << "Enter number of students: ";
    std::cin >> num_students;

    int* student_ids = new int[num_students];

    for (size_t i = 0; i < num_students; i++) {
        student_ids[i] = i;
    }

    delete[] student_ids;

    return 0;
}
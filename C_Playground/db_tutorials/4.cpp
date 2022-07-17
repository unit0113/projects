#include <iostream>
#include <vector>
#include <ctime>


#define NEWSECTION "****************************************"


std::vector<int> range(int, int, int);
void print_pine_tree(int);
void new_section();
double divide(double, double);
void number_guesser();


int main() {

    print_pine_tree(5);

    new_section();

    try {
        divide(5, 0);
    } catch (std::exception &e) {
        std::cout << "Handled Error: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Default Exception" << std::endl;
    }

    new_section();

    number_guesser();
    





    return 0;
}

void new_section() {
    std::cout << NEWSECTION << std::endl;
}


std::vector<int> range(int start, int end, int step) {
    std::vector<int> returnVec;
    for (int i = start; i < end; i += step) {
        returnVec.push_back(i);
    }

    return returnVec;
}


void print_pine_tree(int height) {
    int max_side_size = height - 1;
    char tree = '#';
    char space = ' ';

    // loop through rows
    for (int i = 0; i < height; i++) {
        //print left spaces
        int j { 0 };
        for (j; j < max_side_size - i; j++) {
            std::cout << space;
        }

        //print left side
        for (j; j < max_side_size; j++) {
            std::cout << tree;
        }

        //print center
        std::cout << tree;

        //print right side
        int k { 0 };
        for (k; k < i; k++) {
            std::cout << tree;
        }

        std::cout << std::endl;
    }

    //print stump
    for (int i = 0; i < max_side_size; i++) {
            std::cout << space;
        }
    std::cout << tree << std::endl;
}


double divide(double num1, double num2) {
    if (num2 == 0) {
        throw std::runtime_error("Zero Division Error");
    } else {
        return num1 / num2;
    }
}


void number_guesser() {
    srand(time(NULL));
    int secretNum = std::rand() % 10 + 1;
    int guess {20};
    int guesses {};

    do {
        std::cout << "Guess a number between 1 and 10: ";
        std::cin >> guess;
        guesses++;

        if (guess > secretNum) {
            std::cout << "Too big" << std::endl;
        } else if (guess < secretNum) {
            std::cout << "Too small" << std::endl;
        }
    } while (guess != secretNum);

    std::cout << "Correct! You took " << guesses << " guesses." << std::endl;
}
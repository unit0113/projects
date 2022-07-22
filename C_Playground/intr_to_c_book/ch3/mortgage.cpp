#include <iostream>
#include <math.h>


int main() {
    double mort_amount = 0;
    std::cout << "Enter mortgage amount: ";
    std::cin >> mort_amount;

    int mort_term = 0;
    std::cout << "Enter mortgage term: ";
    std::cin >> mort_term;
    mort_term *= 12;

    double mort_interest = 0;
    std::cout << "Enter mortgage interest: ";
    std::cin >> mort_interest;
    mort_interest /= 12;

    double monthly_payment = (mort_amount * mort_interest * pow((1 + mort_interest), mort_term)) / (pow((1 + mort_interest), mort_term) - 1.0);
    std::cout << "Monthly Payment: " << monthly_payment <<'\n';

    return 0;
}
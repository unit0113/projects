#include <iostream>


#define COUNTY_TAX .05
#define STATE_TAX .04
#define CENTS 100


int main() {
    int revenue {};
    std::string month = "";
    int total_revenue {};
    double total_tax {};
    double county_tax {};
    double state_tax {};


    while (true) {
        std::cout << "Enter Total Revenue (-1 to quit): ";
        std::cin >> revenue;

        if (revenue == -1) {
            break;
        }

        std::cout << "Enter Month: ";
        std::cin >> month;

        total_revenue += revenue;
        std::cout << "Total Revenue: " << total_revenue << '\n';

        county_tax = (revenue * 100 * COUNTY_TAX) / 100.0;
        state_tax = (revenue * 100 * STATE_TAX) / 100.0;
        total_tax += county_tax + state_tax;
        std::cout << "Gross: " << revenue - county_tax - state_tax << '\n';
        std::cout << "County Sales Taxes: " << county_tax << '\n';
        std::cout << "State Sales Taxes: " << state_tax << '\n';
        std::cout << "Total Sales Taxes: " << total_tax << '\n';
        std::cout << '\n';

    }










    return 0;
}
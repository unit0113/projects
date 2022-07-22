#include <iostream>
#include <string>


uint64_t grow_pop(uint64_t population, double growth_rate, int years);
std::string format(uint64_t number);


int main() {

    uint64_t population = 7962375900;
    double growth_rate = .011;

    std::cout << "Current World Population: " << format(population) << std::endl;
    std::cout << "Estimated Population, 1 year: " << format(grow_pop(population, growth_rate, 1)) << std::endl;
    std::cout << "Estimated Population, 3 years: " << format(grow_pop(population, growth_rate, 3)) << std::endl;
    std::cout << "Estimated Population, 5 years: " << format(grow_pop(population, growth_rate, 5)) << std::endl;
    std::cout << "Estimated Population, 10 years: " << format(grow_pop(population, growth_rate, 10)) << std::endl;
    std::cout << "Estimated Population, 25 years: " << format(grow_pop(population, growth_rate, 25)) << std::endl;

    return 0;
}


uint64_t grow_pop(uint64_t population, double growth_rate, int years) {
    uint64_t new_pop = population;
    for (size_t i = 0; i < years; i++)  {
        new_pop *= 1 + growth_rate;
    }

    return new_pop;
}


std::string format(uint64_t number) {
    std::string raw_string = std::to_string(number);
    std::string formatted_string = "";

    size_t comma_counter = raw_string.length() % 3;
    size_t index = 0;
    do {
        formatted_string += raw_string[index];
        comma_counter--;
        if (comma_counter <= 0 && index != raw_string.length() - 1) {
            comma_counter = 3;
            formatted_string += ",";
        }
        index++;
    } while (index < raw_string.length());

    return formatted_string;
}
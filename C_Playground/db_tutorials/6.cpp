#include <iostream>
#include <vector>
#include <sstream>


std::vector<std::string> string_to_vector(std::string input_string, char separator);
std::string vector_to_string(std::vector<std::string> v_customers, char separator);
std::string trim_whitespace(std::string input_string);
std::vector<int> find_substring_matches(std::string phrase, std::string substring);
std::string replace_substrings(std::string phrase, std::string substring_to_replace, std::string substring_to_insert);


int main() {
    std::vector<std::string> vec = string_to_vector("This is a random string", ' ');

    for (size_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i] << std::endl;
    }


    std::vector<std::string> v_customers(3);
    v_customers[0] = "Bob";
    v_customers[1] = "Sue";
    v_customers[2] = "Tom";

    std::string s_customers = vector_to_string(v_customers, ' ');
    std::cout << s_customers << std::endl;


    std::cout << trim_whitespace("        Lot's o' whitespace!!!           ") << std::endl;


    std::string phrase = "To be or not to be";
    std::vector<int> matches = find_substring_matches(phrase, "be");

    for (auto x: matches) {
        std::cout << x << std::endl;
    }


    std::cout << replace_substrings("To be or not to be", "be", "Kobe") << std::endl;


    // ----- 6. CHARACTER FUNCTIONS -----
    char letterZ = 'z';
    char num3 = '3';
    char aSpace = ' ';
    
    std::cout << "Is z a letter or number " << isalnum(letterZ) << "\n";
    std::cout << "Is z a letter " << isalpha(letterZ) << "\n";
    std::cout << "Is z uppercase " << isupper(letterZ) << "\n";
    std::cout << "Is z lowercase " << islower(letterZ) << "\n";
    std::cout << "Is 3 a number " << isdigit(num3) << "\n";
    std::cout << "Is space a space " << isspace(aSpace) << "\n";
    
    // ----- END CHARACTER FUNCTIONS -----


    return 0;
}


std::vector<std::string> string_to_vector(std::string input_string, char separator) {
    std::vector<std::string> word_vec;

    std::stringstream ss(input_string);
    std::string temp_str;
    while(getline(ss, temp_str, separator)) {
        word_vec.push_back(temp_str);
    }

    return word_vec;
}


std::string vector_to_string(std::vector<std::string> v_customers, char separator) {
    std::string word_string = "";

    for (auto cust: v_customers) {
        word_string += cust + separator;
    }

    return trim_whitespace(word_string);
}


std::string trim_whitespace(std::string input_string) {
    static const std::string whitespaces = " \t\f\n\r";
    input_string.erase(input_string.find_last_not_of(whitespaces) + 1);
    input_string.erase(0, input_string.find_first_not_of(whitespaces));
    return input_string;
}


std::vector<int> find_substring_matches(std::string phrase, std::string substring) {
    std::vector<int> matches;
    int index = phrase.find(substring, 0);
    while (index != std::string::npos) {
        matches.push_back(index);
        index = phrase.find(substring, index + 1);
    }

    return matches;
}

std::string replace_substrings(std::string phrase, std::string substring_to_replace, std::string substring_to_insert) {
    std::vector<std::string> word_vec = string_to_vector(phrase, ' ');
    for (size_t i = 0; i < word_vec.size(); i++) {
        if (word_vec[i] == substring_to_replace) {
            word_vec[i] = substring_to_insert;
        }
    }

    return vector_to_string(word_vec, ' ');
}
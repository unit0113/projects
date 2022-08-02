#include <iostream>
#include <vector>
#include <sstream>


std::vector<std::string> tokenize(std::string phrase);
std::vector<std::string> reverse(std::vector<std::string> word_vec);

int main() {

    std::string phrase = "I am always moving forward";
    std::vector<std::string> word_vec = tokenize(phrase);
    std::vector<std::string> reversed_word_vec = reverse(word_vec);

    for (std::string s: reversed_word_vec) {
        std::cout << s << ' ';
    }
    std::cout << std:: endl;

    return 0;
}


std::vector<std::string> tokenize(std::string phrase) {
    std::stringstream ss(phrase);
    std::string temp = "";
    std::vector<std::string> word_vec;

    while(std::getline(ss, temp, ' ')) {
        word_vec.push_back(temp);
    }

    return word_vec;
}


std::vector<std::string> reverse(std::vector<std::string> word_vec) {
    std::vector<std::string> reversed_word_vec;

    for (int i = word_vec.size() - 1; i >= 0; i--) {
        reversed_word_vec.push_back(word_vec[i]);
    }

    return reversed_word_vec;
}
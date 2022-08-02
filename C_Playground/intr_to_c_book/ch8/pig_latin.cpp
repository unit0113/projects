#include <iostream>
#include <vector>
#include <sstream>


std::vector<std::string> tokenize(std::string phrase);
std::vector<std::string> pig_latin(std::vector<std::string> word_vec);


int main() {

    std::string phrase = "Stop mentioning walks";

    std::vector<std::string> word_vec = tokenize(phrase);
    std::vector<std::string> pig_latin_word_vec = pig_latin(word_vec);

    for (std::string s: pig_latin_word_vec) {
        std::cout << s << ' ';
    }
    std::cout << std::endl;

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


std::vector<std::string> pig_latin(std::vector<std::string> word_vec) {
    std::vector<std::string> pig_latin_word_vec;
    for (std::string s: word_vec) {
        s = s.substr(1) + (char)tolower(s[0]) + "ay";
        pig_latin_word_vec.push_back(s);
    }
    return pig_latin_word_vec;
}
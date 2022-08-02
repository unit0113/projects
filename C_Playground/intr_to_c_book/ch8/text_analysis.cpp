#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <array>
#include <map>


std::vector<std::string> tokenize(const std::string phrase);
void letter_count(const std::string phrase);
void print_letter_arr(const std::array<int, 26> &arr);
void word_length(const std::string phrase);
void print_word_length_arr(std::map<int, int> &arr, int max);
void word_count(const std::string phrase);
void print_word_count_arr(const std::map<std::string, int> &arr);


int main() {

    std::string phrase = "To be or not to be, that is the question. Whether 'tis nobler in the mind \
                          to suffer the slings and arrows of outrageous fortune, or to take Arms against a Sea of troubles, \
                          and by opposing end them: to die, to sleep.";

    letter_count(phrase);
    word_length(phrase);
    word_count(phrase);

    return 0;
}


std::vector<std::string> tokenize(const std::string phrase) {
    std::string no_punc_phrase = "";
    for (char c: phrase) {
        if (!ispunct(c)) {
            no_punc_phrase += tolower(c);
        }
    }

    std::stringstream ss(no_punc_phrase);
    std::string temp = "";
    std::vector<std::string> word_vec;

    while(std::getline(ss, temp, ' ')) {
        word_vec.push_back(temp);
    }

    return word_vec;
}


void letter_count(const std::string phrase) {
    std::array<int, 26> letter_arr = {};
    std::string phrase_lower = phrase;
    std::transform(phrase_lower.begin(), phrase_lower.end(), phrase_lower.begin(), ::tolower);

    for (char c: phrase_lower) {
        if (isalpha(c)) {
            letter_arr[c - 'a']++;
        }
    }

    print_letter_arr(letter_arr);
}


void print_letter_arr(const std::array<int, 26> &arr) {
    // Print top dashes
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << "------";
    }
    std::cout << '-' << std::endl;

    // Print indexes
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << "| " << std::setw(2) << (char)(i + 'a') << "  ";
    }
    std::cout << '|' << std::endl;   

    // Print mid dashes
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << "------";
    }
    std::cout << '-' << std::endl; 

    // Print items
    for (int x: arr) {
        std::cout << "| " << std::setw(2) << x << "  ";
    }
    std::cout << '|' << std::endl;

    // Print bottom dashes
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << "------";
    }
    std::cout << '-' << std::endl;
}


void word_length(const std::string phrase) {
    std::vector<std::string> word_vec = tokenize(phrase);
    std::map<int, int> word_map;
    int max = 0;
    int word_len = 0;

    for (std::string s: word_vec) {
        word_len = s.size();
        max = std::max(word_len, max);
        if (word_map.contains(word_len)) {
            word_map[word_len]++;
        } else {
            word_map[word_len] = 1;
        }
    }

    print_word_length_arr(word_map, max);
}


void print_word_length_arr(std::map<int, int> &arr, int max) {
    // Print top dashes
    for (size_t i = 1; i <= max; i++) {
        std::cout << "------";
    }
    std::cout << '-' << std::endl;

    // Print indexes
    for (size_t i = 1; i <= max; i++) {
        std::cout << "| " << std::setw(2) << i << "  ";
    }
    std::cout << '|' << std::endl;   

    // Print mid dashes
    for (size_t i = 1; i <= max; i++) {
        std::cout << "------";
    }
    std::cout << '-' << std::endl; 

    // Print items
    for (size_t i = 1; i <= max; i++) {
        if (arr.contains(i)) {
            std::cout << "| " << std::setw(2) << arr[i] << "  ";
        } else {
            std::cout << "|  0  ";
        }
    }
    std::cout << '|' << std::endl;

    // Print bottom dashes
    for (size_t i = 1; i <=max; i++) {
        std::cout << "------";
    }
    std::cout << '-' << std::endl;
}


void word_count(const std::string phrase) {
    std::vector<std::string> word_vec = tokenize(phrase);
    std::map<std::string, int> word_map;

    for (std::string s: word_vec) {
        if (word_map.contains(s)) {
            word_map[s]++;
        } else {
            word_map[s] = 1;
        }
    }
    print_word_count_arr(word_map);
}


void print_word_count_arr(const std::map<std::string, int> &arr) {
    // Print words
    for (auto pair: arr) {
        std::cout << pair.first << '\t';
    }
    std::cout << std::endl;   

    // Print items
    for (auto pair: arr) {
        for (size_t i = 0; i < pair.first.size() / 2; i++) {
            std::cout << ' ';
        }
        std::cout << pair.second << '\t';
        if (pair.first.size() >= 8) {
            std::cout << '\t';
        }
    }
    std::cout << std::endl;
}
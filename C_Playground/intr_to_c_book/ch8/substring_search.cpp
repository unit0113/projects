#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>


std::vector<std::string> tokenize(std::string phrase);
int substring_count(std::string phrase, std::string substring);
int substring_first_occurance(std::string phrase, std::string substring);


int main() {

    std::string phrase = "I am always moving forward, never back. The past shapes our identity, but our future is forward.";
    std::vector<std::string> word_vec = tokenize(phrase);
    std::string substring = "forward";

    std::cout << "The seed phrase contains 'forward' " << substring_count(phrase, substring) << " times." << std::endl;
    std::cout << "The first occurance of 'forward' is at index " << substring_first_occurance(phrase, substring) << '.' << std::endl;

    return 0;
}


std::vector<std::string> tokenize(std::string phrase) {
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


int substring_count(std::string phrase, std::string substring) {
    std::vector<std::string> tokens = tokenize(phrase);

    int count = 0;
    std::string substring_lower = substring;
    std::transform(substring_lower.begin(), substring_lower.end(), substring_lower.begin(), ::tolower);

    for (std::string s: tokens) {
        if (s.compare(substring) == 0) {
            count++;
        }
    }
    return count;
}


int substring_first_occurance(std::string phrase, std::string substring) {
    int index = 0;
    bool match = true;

    while (index <= phrase.size() - substring.size()) {
        match = true;
        if (phrase[index] == substring[0]) {
            for (size_t i = 1; i < substring.size(); i++) {
                if (!(phrase[index + i] == substring[i])) {
                    match = false;
                    break;
                }
            }

            if (match) {
                return index;
            }

        } else {
            index++;
        }
    } 

    return -1;
}
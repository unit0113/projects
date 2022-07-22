#include <iostream>
#include <vector>
#include <string>


std::string encrypt(std::string input);
std::string decrypt(std::string input);
void swap(std::vector<int> &input);


int main() {
    std::string input = "";
    std::cout << "Enter a number to encrypt: ";
    std::cin >> input;

    std::string encrypted_input = encrypt(input);
    std::cout << "Encrypted:\t" << encrypted_input << std::endl;
    std::cout << "Decrypted:\t" << decrypt(encrypted_input) << std::endl;

    return 0;
}


std::string encrypt(std::string input) {
    std::vector<int> input_vec;
    int temp = 0;
    for (size_t i = 0; i < input.length(); i++) {
        temp = input[i] - '0';
        input_vec.push_back((7 + temp) % 10);
    }

    swap(input_vec);

    std::string output_str = "";
    for (auto x: input_vec) {
        output_str += std::to_string(x);
    }

    return output_str;
}


std::string decrypt(std::string input) {
    std::vector<int> input_vec;
    int temp = 0;
    for (size_t i = 0; i < input.length(); i++) {
        temp = input[i] - '0';
        input_vec.push_back(temp);
    }

    swap(input_vec);

    std::string output_str = "";
    for (auto x: input_vec) {
        temp = (x - 7);
        output_str += std::to_string(temp > 0 ? temp : temp + 10);
    }

    return output_str;
}


void swap(std::vector<int> &input) {
    int temp = input[0];
    input[0] = input[2];
    input[2] = temp;

    temp = input[1];
    input[1] = input[3];
    input[3] = temp;
}

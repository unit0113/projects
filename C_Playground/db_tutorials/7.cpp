#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <math.h>


std::string caesar_encrypt(std::string s_input, int key);
std::string caesar_decrypt(std::string s_encrypted_input, int key);
std::vector<std::string> string_to_vector(std::string input_string, char separator);
void solve_for_x(std::string equation);
bool is_prime(int num);
std::vector<int> get_primes(int max_prime);
std::vector<int> gen_random_vec(int num_values, int min_val, int max_val);


int main() {

    // Caesar
    int key = 15;
    std::string s_input = "Make me secret";
    std::string s_encrypted_input = caesar_encrypt(s_input, key);
    std::string s_decrypted_input = caesar_decrypt(s_encrypted_input, key);

    std::cout << "Input: " << s_input << std::endl;
    std::cout << "Encrypted: " << s_encrypted_input << std::endl;
    std::cout << "Decrypted: " << s_decrypted_input << std::endl;


    // Solve for X
    std::cout << "Enter an equation to solve: ";
    std::string equation = "";
    getline(std::cin, equation);
    solve_for_x(equation);


    // List of primes
    int num = 0;
    std::cout << "Enter a number to check: ";
    std::cin >> num;
    std::cout.setf(std::ios::boolalpha);
    std::cout << "Is " << num << " prime: " << is_prime(num) << std::endl;

    std::cout << "Generate primes up to: ";
    int max_prime = 0;
    std::cin >> max_prime;
    std::vector<int> vec_primes = get_primes(max_prime);
    for (int x: vec_primes) {
        std::cout << x << std::endl;
    }


    // Vec of random numbers
    std::vector<int> vec_randoms = gen_random_vec(10, 5, 50);
    for (int x: vec_randoms) {
        std::cout << x << std::endl;
    }


    return 0;
}


std::string caesar_encrypt(std::string s_input, int key) {
    std::string s_encrypted_input = "";
    int char_code = 0;
    char letter;
    for (char c: s_input) {
        if (isalpha(c)) {
            char_code = (int)c + key;

            if (isupper(c)) {
                if (char_code > (int)'Z') {
                    char_code -= 26;
                }
            } else {
                 if (char_code > (int)'z') {
                    char_code -= 26;
                }
            }

            letter = char_code;
            s_encrypted_input += letter;

        } else {
            s_encrypted_input += c;
        }     
    }
    return s_encrypted_input;
}


std::string caesar_decrypt(std::string s_encrypted_input, int key) {
    std::string s_decrypted_output = "";
    int char_code = 0;
    char letter;
    for (char c: s_encrypted_input) {
        if (isalpha(c)) {
            char_code = (int)c - key;

            if (isupper(c)) {
                if (char_code < (int)'A') {
                    char_code += 26;
                }
            } else {
                 if (char_code < (int)'a') {
                    char_code += 26;
                }
            }

            letter = char_code;
            s_decrypted_output += letter;

        } else {
            s_decrypted_output += c;
        }     
    }
    return s_decrypted_output;
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


void solve_for_x(std::string equation) {
    std::vector<std::string> vec_equation = string_to_vector(equation, ' ');
    int num1 = std::stoi(vec_equation[2]);
    int num2 = std::stoi(vec_equation[4]);
    std::cout << "X = " << num2 - num1 << std::endl;
}


bool is_prime(int num) {
    for (size_t i = 2; i < sqrt(num); i++) {
        if ((num % i) == 0) {
            return false;
        }
    }

    return true;
}


std::vector<int> get_primes(int max_prime) {
    std::vector<int> primes;
    for (size_t i = 1; i <= max_prime; i++) {
        if (is_prime(i)) {
            primes.push_back(i);
        }
    }

    return primes;
}


std::vector<int> gen_random_vec(int num_values, int min_val, int max_val) {
    std::vector<int> vec_random;
    srand(time(NULL));
    int range = 1 + max_val - min_val;
    for (size_t i = 0; i < num_values; i++) {
        vec_random.push_back(rand() % range + min_val);
    }
    return vec_random;
}
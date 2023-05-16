/*
    Suffix count

    In this problem, you will write a function that returns the count of a string’s suffix in a given string, S. 
    A suffix is a combination of one or more symbols appended at the end of the string. 
    You will be given the length, L of the suffix as a parameter to the function

    Sample Input:
        et tu, brute
        1
    
    Sample Output:
        2 
*/

#include <iostream>
#include <string>

int suffixCount(std::string S, int L) {
    // Your code here
    int count = 0;
    std::string::size_type curr_pos {};
    std::string suffix = S.substr(S.length() - L);
    while ((curr_pos = S.find(suffix, curr_pos)) != std::string::npos) {
        ++count;
        curr_pos += 1;
    }

    return count;
}

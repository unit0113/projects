/*
    Are permutations

    In this problem, you are given an array of strings strs such that each string only consists of lowercase letters. 
    Two strings strs[i] and strs[j] are permutations of each other if all the characters of strs[j] can be rearranged
    such that strs[i] == strs[j]. Write a method that returns true if all the strings in strs are permutations of each other.

    Example 1:
        Input: strs = ["abba", "bbaa", "aabb"]
        Output: true
        Explanation: Each string can be rearranged to match another

    Example 2:
        Input: strs = ["abc", "abbc"]
        Output: false
        Explanation: “abc” cannot be rearranged to “abbc” because it only has one ‘b’

    Example 3:
        Input: strs = ["gator", "rotag", "sator"]
        Output: false
    
    Example 4:
        Input: strs = ["shoes", "sheso", "hesos", "shooes", "shoe"]
        Output: false

    Example 5:
        Input: strs = [""]
        Output: true
*/

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

bool arePermutations(std::vector<std::string> strs) {
    // Your code here
    int n = strs[0].length();
    std::string first = strs[0];
    std::sort(first.begin(), first.end());
    for (std::string str: strs) {
        std::sort(str.begin(), str.end());
        if (str.length() != n || str != first) {
            return false;
        }
    }

    return true;
}
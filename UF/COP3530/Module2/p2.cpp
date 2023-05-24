#include <iostream>
#include <unordered_set>
#include <fstream>
#include <array>
#include <string>
using namespace std;


// 12 book titles, loading into 3x4 array, 2 are repeats
const int n = 3;
const int m = 4;


unordered_set<string> getUnique(const array<array<string, m>, n> &books) {
    // Initialize set and reserve space
    unordered_set<string> unique_books;
    unique_books.reserve(n * m);

    // Insert book titles into set
    for (int i{}; i < n; ++i) {
        for (int j{}; j < m; ++j) {
            unique_books.insert(books[i][j]);
            cout << unique_books.bucket_count() << endl;
        }
    }

    return unique_books;
}


int main() {
    // Load book titles into 2D array
    array<array<string, m>, n> books;

    // Fill array with book titles
    ifstream file("p2data_worst_case.txt");
    string text;
    for (int i{}; i < n; ++i) {
        for (int j{}; j < m; ++j) {
            getline(file, text);
            books[i][j] = text;
        }
    }

    // Hash book titles and return only unique
    unordered_set<string> unique_books = getUnique(books);
    
    // Print total and titles
    cout << "There are " << unique_books.size() << " unique book titles in the data" << endl;
    for (const auto& book: unique_books) {
        cout << book << endl;
    }
}
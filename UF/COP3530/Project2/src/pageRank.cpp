#include <iostream>
#include <sstream>
#include "pageRank.h"

using namespace std;


void PageRank::run() {
	/*
	*	Main runner method
	*	Gets and stores input, iterates through page rank algorithm
	*/

    int power = parseInput();
    

}

int PageRank::parseInput() {
    // Get number of commands
	int num_commands{};
	cin >> num_commands;

    // Get power iterations
    int power{};
    cin >> power;
	
	// Clear new line character
	cin.ignore();

	// Read and insert pages into adjacency list
	string line, fromPage, toPage;
    int splitPos{};

	for (int i{}; i < num_commands; ++i) {
		// Tokenize input
		getline(cin, line);
		istringstream ss(line);
        splitPos = line.find(' ');
		fromPage = line.substr(0, splitPos);
        toPage = line.substr(splitPos + 1, line.size() - 1);
        adjacencyList.insert(fromPage, toPage);
	}

    return power;
}

void PageRank::iteratePageRank(int power) {
    // Set initial values of page rank
    double initVal = 1 / adjacencyList.size();
    
}
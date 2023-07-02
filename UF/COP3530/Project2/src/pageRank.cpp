#include <iostream>
#include <iomanip>
#include <sstream>
#include "pageRank.h"

using namespace std;

void PageRank::run() {
	/*
	*	Main runner method
	*	Gets and stores input, iterates through page rank algorithm
	*/

    int power = parseInput();
    map<string, double> ranks = iteratePageRank(power);
	printResults(ranks);

}

int PageRank::parseInput() {
	/*
	*	Parses user input and adds values to adjacency list
	*	First line of user input is two values seperated by a space
	*	The first value is the number of edges being inserted
	*	The second value is the power iteration value
	*	Follow on lines consist of two values, 
	*	a fromPage and a toPage, separated by a space
	*/

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

map<string, double> PageRank::iteratePageRank(int power) {
	/*
	*	Page rank algorithm
	*	Calculates initial values and performs the specified number of iterations
	*/

    // Set initial values of page rank
    double initVal = 1.0f / adjacencyList.size();

	// Initilize page ranks
	map<string, double> ranks;
	for (const auto& kv: adjacencyList) {
		ranks[kv.first] = initVal;
	}

	// Perform power iterations
	map<string, double> ranks_old;
	int outDegree{};
	// Power iteration starts at 1, p = 2 means 1 iteration
	for (size_t i=1; i < power; ++i) {
		// Save current rank values
		ranks_old = ranks;
		// Zero out previous values
		for (auto& kv: ranks) {
			ranks[kv.first] = 0;
		}

		// Perform summation
		for (const auto& kv: adjacencyList) {
			outDegree = kv.second.size();
			for (const auto& toPage: kv.second) {
				ranks[toPage] += ranks_old[kv.first] / outDegree;
			}
		}
	}
	return ranks;
}

void PageRank::printResults(map<string, double> ranks) const {
	/*
	*	Prints page rank results to two decimal places
	*/

	// Two decimal places
	cout << fixed << setprecision(2);
	for (const auto& kv: ranks) {
		cout << kv.first << ' ' << kv.second << endl;
	}
}
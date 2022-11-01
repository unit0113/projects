#include <iostream>
#include <map>
#include <random>
#include <ctime>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iterator>
using namespace std;

mt19937 random_mt;

struct StateData {
    int m_capita;
    int m_pop;
    int m_income;
    int m_households;

    StateData(int capita, int pop, int income, int households)
        : m_capita(capita), m_pop(pop), m_income(income), m_households(households) {}

    void print() const {
        cout << "Population: " << m_pop << endl;
        cout << "Per Capita Income: " << m_capita << endl;
        cout << "Median Household Income: " << m_income << endl;
        cout << "Number of Households: " << m_households << endl;
    }
};

int Random(int min, int max) {
	uniform_int_distribution<int> dist(min, max);
	return dist(random_mt);
}

void initializeMap(map<int, int>& rollResults, int max);
void printMap(map<int, int>& rollResults);
map<string, StateData> loadFile();
void printAllStates(map<string, StateData>& states);
void printState(map<string, StateData>& states, string state);

int main() {
	cout << "1. Random Numbers\n";
	cout << "2. State Info\n";

	int option;
	cin >> option;

	if (option == 1) {
		int randomSeed;
		cout << "Random seed value: ";
		cin >> randomSeed;
		random_mt.seed(randomSeed);


		cout << "Number of times to roll the die: ";
        int rolls;
        cin >> rolls;

		cout << "Number of sides on this die: ";
        int sides;
        cin >> sides;
        cout << endl;

		// Roll Dice
        map<int, int> rollResults;
        initializeMap(rollResults, sides);

        for (int i{}; i < rolls; ++i) {
            ++rollResults[Random(1, sides)];
        }

        printMap(rollResults);

	} else if (option == 2) {
	    // Load the states
        map<string, StateData> states = loadFile();
	   
	    // Get input for option 1 (show all states) or 2 (do a search for a particular state)
        cout << "1. Print all states\n" << "2. Search for a state\n";
        cin >> option;
        
        if (option == 1) {
            printAllStates(states);
        } else if (option == 2) {
            string state;
            getchar();
            getline(cin, state);
            printState(states, state);
        } else {
            throw std::runtime_error("Invalid selection");
        }
	}

	return 0;
}

void initializeMap(map<int, int>& rollResults, int max) {
    for (int i = 1; i <= max; ++i) {
        rollResults[i] = 0;
    }
}


void printMap(map<int, int>& rollResults) {
    map<int, int>::iterator it = rollResults.begin();
    while (it != rollResults.end()) {
        cout << it->first << ": " << it->second << endl;
        ++it;
    }
}


map<string, StateData> loadFile() {
    ifstream file("states.csv");
	map<string, StateData> states;

	// Check for file
    if (!file) throw std::runtime_error("File not found");

    // Loop through file
    int capita, pop, income, households;
    string inLine, state, token;
    istringstream inStream;

    getline(file, inLine);      // Skip first line
    while (getline(file, inLine)) {
        inStream.clear();
        inStream.str(inLine);

        // Parse Line
        getline(inStream, state, ',');
        getline(inStream, token, ',');
        capita = stoi(token);
        getline(inStream, token, ',');
        pop = stoi(token);
        getline(inStream, token, ',');
        income = stoi(token);
        getline(inStream, token, ',');
        households = stoi(token);
        states.emplace(state, StateData(capita, pop, income, households));

    }
    return states;
}


void printAllStates(map<string, StateData>& states) {
    map<string, StateData>::iterator it = states.begin();
    while (it != states.end()) {
        cout << it->first << endl;
        it->second.print();
        ++it;
    }
}


void printState(map<string, StateData>& states, string state) {
    map<string, StateData>::iterator it = states.find(state);
    if (it != states.end()) {
        cout << it->first << endl;
        it->second.print();
    } else {
        cout << "No match found for " << state << endl;
    }
}
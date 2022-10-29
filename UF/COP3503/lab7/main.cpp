#include <iostream>
#include <map>
#include <random>
#include <ctime>
using namespace std;

mt19937 random_mt;

int Random(int min, int max) {
	uniform_int_distribution<int> dist(min, max);
	return dist(random_mt);
}

void initializeMap(map<int, int>& rollResults, int max);
void printMap(map<int, int>& rollResults);

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
	   
	   // Get input for option 1 (show all states) or 2 (do a search for a particular state)

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
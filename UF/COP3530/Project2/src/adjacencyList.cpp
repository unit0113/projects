#include "adjacencyList.h"
using namespace std;


void AdjacencyList::insert(std::string& fromPage, std::string& toPage) {
	/*
	*	Modified insert function to ensure all keys exist in graph
	*/

    // Check if key exists
    if (adjacencyList.find(fromPage) == adjacencyList.end()) {
        // Intialize new list with toPage as member
        adjacencyList.insert({fromPage, list<string> {toPage}});
    } else {
        adjacencyList[fromPage].push_back(toPage);
    }
    //Check if key exists for toPage, if not, initialize
    if (adjacencyList.find(toPage) == adjacencyList.end()) {
        // Intialize new empty list
        adjacencyList.insert({toPage, list<string> {}});
    }
}
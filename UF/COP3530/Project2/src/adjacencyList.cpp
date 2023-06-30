#include "adjacencyList.h"
using namespace std;


void AdjacencyList::insert(std::string& fromPage, std::string& toPage) {
    // Check if key exists
    if (adjacencyList.find(fromPage) == adjacencyList.end()) {
        // Intialize new vector with toPage as member
        adjacencyList.insert({fromPage, vector<string> {toPage}});
    } else {
        adjacencyList[fromPage].push_back(toPage);
    }
}

int AdjacencyList::size() const {
    return adjacencyList.size();
}
#include "adjacencyList.h"
using namespace std;


void AdjacencyList::insert(const std::string& fromPage, const std::string& toPage) {
    // Check if key exists
    if (adjacencyList.find(fromPage) == adjacencyList.end()) {
        // Intialize new vector with toPage as member
        adjacencyList.insert({fromPage, vector<const string> {toPage}});
    } else {
        adjacencyList[fromPage].push_back(toPage);
    }
}
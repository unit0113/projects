/*
    Not so simple Graph

    Write C++ code for implementing a Graph data structure that supports a directed graph with 
    self-loops and parallel edges. You are expected to implement the following methods and a main
    method is already built for you:

        void *insertEdge*(int from, int to, int weight);   // 1
        bool *isEdge*(int from, int to);                   // 2
        int *sumEdge*(int from, int to);                   // 3
        vector<int> *getWeight*(int from, int to);         // 4
        vector<int> *getAdjacent*(int vertex);             // 5


    Sample Input:
        7    
        1 0 0 10  
        1 0 1 20 
        1 0 2 30
        2 0 0  
        3 0 2 
        4 0 1
        5 0
    
    Sample Output:
        1  
        30
        20 
        0 1 2
*/

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <list>
#include <algorithm>


class Graph {
   private:
      std::unordered_map<int, std::list<std::pair<int, int>>> adjacencyList;

    public:
      void insertEdge(int from, int to, int weight);  
      bool isEdge(int from, int to);  
      int sumEdge(int from, int to); 
      std::vector<int> getWeight(int from, int to); 
      std::vector<int> getAdjacent(int vertex); 
};

void Graph::insertEdge(int from, int to, int weight) {
   /*
        TODO: insertEdge() adds a new edge between the from 
        and to vertex.
   */
    // Check if key exists
    if (adjacencyList.find(from) == adjacencyList.end()) {
        // Intialize new list with toPage as member
        adjacencyList.insert({from, std::list<std::pair<int, int>>{}});
    }
    
    adjacencyList[from].push_back(std::make_pair(to, weight));

    //Check if key exists for toPage, if not, initialize
    if (adjacencyList.find(to) == adjacencyList.end()) {
        // Intialize new empty list
        adjacencyList.insert({to, std::list<std::pair<int, int>>{}});
    }

}
        
bool Graph::isEdge(int from, int to) {
    /*
        TODO: isEdge() returns a boolean indicating true 
        if there is an edge between the from and to vertex
    */

    if (adjacencyList.find(from) == adjacencyList.end()) {return false;}
    for (const auto& p : adjacencyList[from]) {
            if (p.first == to) {return true;}
        }

    return false;
}

int Graph::sumEdge(int from, int to) {
    /*
        TODO: sumEdge() returns the sum of weights of all edges 
        connecting the from and to vertex. Returns 0 if no edges 
        connect the two vertices.
    */

    int sum{};
    if (isEdge(from, to)) {
        for (const auto& p : adjacencyList[from]) {
            if (p.first == to) {sum += p.second;}
        }
    }

    return sum;
}

std::vector<int> Graph::getWeight(int from, int to) {
    /*
        TODO: getWeight() returns a sorted vector containing all 
        weights of the edges connecting the from and to vertex
    */
    
    std::vector<int> weights;
    if (isEdge(from, to)) {
        for (const auto& p : adjacencyList[from]) {
            if (p.first == to) {weights.push_back(p.second);}
        }
    }

    return weights;
}

std::vector<int> Graph::getAdjacent(int vertex) {
    /*
        TODO: getAdjacent() returns a sorted vector of all vertices
        that are connected to a vertex
    */
    
    std::vector<int> neighbors;
    for (const auto& p : adjacencyList[vertex]) {
        neighbors.push_back(p.first);
    }
    std::sort(neighbors.begin(), neighbors.end());

    return neighbors;
}

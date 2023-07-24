/*
    Maze Escape

    Sometimes when dealing with graphs, it is easiest to leave it in its input format 
    rather than creating another structure like an adjacency list/matrix. You are given 
    a graph represented as a vector of strings. Write a function 
    `traverse(vector<string>& graph)` that takes as input a graph and returns the length 
    of the shortest path between vertices `s` and `t`. If no path exists between `s` and 
    `t`, then return `-1`. The details of the graph as a vector of strings are as follows:

    1. The start position is represented by a 's' and will always be the first character of 
       the first string (`graph[0][0]`). 
    2. The end position is represented by a 't' and will always be the last character of the 
       final string (`graph[graph.size()-1][graph[0].length()-1]`).
    3. A '.' character represents a normal vertex in the graph.
    4. A '#' character represents that you cannot visit this vertex in the graph (or there 
       is no vertex at this position).
    5. Adjacent vertices are those immediately horizontal or vertical from the current vertex 
       (diagonal moves are not allowed).
    6. The cost of moving along one edge (from one vertex to an adjacent vertex) is always 1 
       (i.e. this is an unweighted graph).

    Sample Input
        s#.#.   
        .#...  
        ...#t    

    Sample Output: 8
*/

#include <iostream>
#include <vector>
#include <queue>
#include <tuple>


std::vector<std::tuple<int, int, int>> get_neighbors(std::tuple<int, int, int>& position, std::vector<std::string>& graph) {
    std::vector<std::tuple<int, int, int>> neighbors;

    // Right
    if (std::get<1>(position) + 1 < graph[0].size()
        && graph[std::get<0>(position)][std::get<1>(position) + 1] != '#'
        && (std::get<2>(position) + 1 < graph[std::get<0>(position)][std::get<1>(position) + 1]
            || graph[std::get<0>(position)][std::get<1>(position) + 1] == -1)) {
        neighbors.push_back(std::make_tuple(std::get<0>(position), std::get<1>(position) + 1, std::get<2>(position) + 1));
    }

   // Left
    if (std::get<1>(position) - 1 >= 0
        && graph[std::get<0>(position)][std::get<1>(position) - 1] != '#'
        && (std::get<2>(position) + 1 < graph[std::get<0>(position)][std::get<1>(position) - 1]
            || graph[std::get<0>(position)][std::get<1>(position) - 1] == -1)) {
        neighbors.push_back(std::make_tuple(std::get<0>(position), std::get<1>(position) - 1, std::get<2>(position) + 1));
    }

   // Up
    if (std::get<0>(position) - 1 >= 0
        && graph[std::get<0>(position) - 1][std::get<1>(position)] != '#'
        && (std::get<2>(position) + 1 < graph[std::get<0>(position) - 1][std::get<1>(position)]
            || graph[std::get<0>(position) - 1][std::get<1>(position)] == -1)) {
        neighbors.push_back(std::make_tuple(std::get<0>(position) - 1, std::get<1>(position), std::get<2>(position) + 1));
    }

   // Down
    if (std::get<0>(position) + 1 < graph.size()
        && graph[std::get<0>(position) + 1][std::get<1>(position)] != '#'
        && (std::get<2>(position) + 1 < graph[std::get<0>(position) + 1][std::get<1>(position)]
            || graph[std::get<0>(position) + 1][std::get<1>(position)] == -1)) {
        neighbors.push_back(std::make_tuple(std::get<0>(position) + 1, std::get<1>(position), std::get<2>(position) + 1));
    }

    return neighbors;
}


int traverse(std::vector<std::string>& graph) {
    if (graph.size() == 0 || graph[0].size() == 0) {return -1;}

    std::priority_queue<std::tuple<int, int, int> , std::vector<std::tuple<int, int, int>>, std::greater<std::tuple<int, int, int>>> p_q;

    p_q.push(std::make_tuple(0, 0, 0));

    while (!p_q.empty()) {
        std::tuple<int, int, int> position = p_q.top();
        p_q.pop();
        std::vector<std::tuple<int, int, int>> neighbors = get_neighbors(position, graph);
        for (auto& neighbor : neighbors) {
            if (graph[std::get<0>(neighbor)][std::get<1>(neighbor)] == 't') {
                return std::get<2>(neighbor);
            }
            graph[std::get<0>(neighbor)][std::get<1>(neighbor)] = std::get<2>(neighbor);
            p_q.push(neighbor);
        }
    }
    return -1;
}

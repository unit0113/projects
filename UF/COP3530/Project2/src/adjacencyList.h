#pragma once

#include <map>
#include <vector>
#include <string>

class AdjacencyList {
	private:
		std::map<std::string, std::vector<std::string>> adjacencyList;

		// Helper Functions

	public:
		void insert(std::string& fromPage, std::string& toPage);
		int size() const;
};

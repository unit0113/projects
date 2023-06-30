#pragma once

#include <map>
#include <vector>
#include <string>

class AdjacencyList {
	private:
		std::map<const std::string, std::vector<const std::string>> adjacencyList;

		// Helper Functions

	public:
		void insert(const std::string& fromPage, const std::string& toPage);
		
};

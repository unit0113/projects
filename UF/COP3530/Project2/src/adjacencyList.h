#pragma once

#include <map>
#include <list>
#include <string>

class AdjacencyList {
	private:
		std::map<std::string, std::list<std::string>> adjacencyList;
		typedef typename std::map<std::string, std::list<std::string>>::iterator it;

		// Helper Functions

	public:
		void insert(std::string& fromPage, std::string& toPage);
		int size() const {return adjacencyList.size();};
		// Iterator
		it begin() {return adjacencyList.begin();};
		it end() {return adjacencyList.end();};
};

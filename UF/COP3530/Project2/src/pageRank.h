#pragma once

#include "adjacencyList.h"

class PageRank {
	private:
		AdjacencyList adjacencyList;

		// Helper Functions
		int parseInput();
		std::map<std::string, double> iteratePageRank(int power);
		void printResults(std::map<std::string, double> ranks) const;

	public:
		void run();

};
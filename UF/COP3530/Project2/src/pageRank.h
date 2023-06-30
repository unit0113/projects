#pragma once
#include <regex>
#include <sstream>
#include "adjacencyList.h"

class PageRank {
	private:
		AdjacencyList adjacencyList;

		// Helper Functions
		int parseInput();
		void iteratePageRank(int power);

	public:
		void run();

};
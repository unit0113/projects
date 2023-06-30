#pragma once
#include <regex>
#include <sstream>
#include "adjacencyList.h"

class PageRank {
	private:
		AdjacencyList tree;

		// Helper Functions

	public:
		void run();

};
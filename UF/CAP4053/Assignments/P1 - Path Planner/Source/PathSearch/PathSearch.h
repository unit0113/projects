#include "../platform.h" // This file will make exporting DLL symbols simpler for students.
#include "../../Source/Framework/TileSystem/Tile.h"
#include "../../Source/Framework/TileSystem/TileMap.h"
#include "../PriorityQueue.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>

namespace ufl_cap4053
{
	namespace searches
	{
		class PlannerNode {
		public:
			PlannerNode(const Tile* tile, const Tile* parent = nullptr, double gCost = 0, double hCost = 0) : tile(tile), parent(parent), givenCost(gCost), heuristicCost(hCost) { finalCost = givenCost + heuristicCost; };
			bool operator == (const PlannerNode& other) { return other.tile == tile; };
			const Tile* getTile() const { return tile; };
			void setParent(const Tile* newParent) { parent = newParent; };
			const Tile* getParent() const { return parent; };
			double getGivenCost() const { return givenCost; };
			void setGivenCost(double newCost) { givenCost = newCost; };
			double getFinalCost() const { return finalCost; };


		private:
			const Tile* tile;
			const Tile* parent;
			double givenCost;
			double heuristicCost;
			double finalCost;

		};

		class PathSearch {
		public:
			DLLEXPORT PathSearch() : open(greaterThan) {};
			DLLEXPORT ~PathSearch();
			DLLEXPORT void load(TileMap* _tileMap); //Creates search graph
			DLLEXPORT void initialize(int startRow, int startCol, int goalRow, int goalCol); //Called before update of path planner
			DLLEXPORT void update(long timeslice); //Execute path planner
			DLLEXPORT void shutdown(); //Cleans up memory for this search
			DLLEXPORT void unload(); //Cleans up memory for tile map
			DLLEXPORT bool isDone() const; //If solution found
			DLLEXPORT std::vector<Tile const*> const getSolution() const;

		private:
			TileMap* tileMap;
			std::unordered_map<const Tile*, std::unordered_set<const Tile*>> searchGraph;
			std::unordered_map<const Tile*, PlannerNode*> visited;
			const Tile* startTile;
			const Tile* goalTile;
			bool isComplete;
			PriorityQueue<PlannerNode*> open;
			std::vector<Tile const*> solution{};

			void aStarIteration();
			void buildSolution();
			bool areAdjacent(const Tile* lhs, const Tile* rhs) const;
			int heuristic(const Tile* src) const;
			bool greaterThan(PlannerNode* const& lhs, PlannerNode* const& rhs) { return lhs->getFinalCost() > rhs->getFinalCost(); };
		};
	}
}  // close namespace ufl_cap4053::searches

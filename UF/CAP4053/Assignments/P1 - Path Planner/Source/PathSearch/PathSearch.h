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
			PlannerNode(const Tile* tile, const Tile* parent = nullptr, double gCost = 0, double hCost = 0);
			bool operator == (const PlannerNode& other) { return other.tile == tile; };
			const Tile* getTile() const { return tile; };
			const std::pair<int, int> getTileCoords() const;
			void setParent(const Tile* newParent) { parent = newParent; };
			const Tile* getParent() const { return parent; };
			double getGivenCost() const { return givenCost; };
			void setGivenCost(double newCost);
			double getFinalCost() const { return finalCost; };


		private:
			const Tile* tile;
			const Tile* parent;
			double givenCost;
			double heuristicCost;
			double finalCost;

		};

		// Non-class helper functions
		bool greaterThan(PlannerNode* const &lhs, PlannerNode* const &rhs) { return lhs->getFinalCost() > rhs->getFinalCost(); };
		struct tile_hash {
			std::size_t operator()(const Tile* t) const {
				return std::hash<int>{}(t->getRow()) ^ std::hash<int>{}(t->getColumn());
			}
		};
		struct pair_hash {
			std::size_t operator()(const std::pair<int, int> &p) const {
				return std::hash<int>{}(p.first) ^ std::hash<int>{}(p.second);
			}
		};

		class PathSearch {
		public:
			DLLEXPORT PathSearch();
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
			std::unordered_map<const Tile*, std::unordered_set<const Tile*, tile_hash>, tile_hash> searchGraph;
			std::unordered_map<const Tile*, PlannerNode*, tile_hash> visited;
			const Tile* startTile;
			const Tile* goalTile;
			bool isComplete;
			PriorityQueue<PlannerNode*> open;
			std::vector<Tile const*> solution{};

			void aStarIteration();
			void buildSolution();
			bool areAdjacent(const Tile* lhs, const Tile* rhs) const;
			double heuristic(const Tile* src) const;
			bool isInvalidNode(PlannerNode* current) const;
		};
	}
}  // close namespace ufl_cap4053::searches

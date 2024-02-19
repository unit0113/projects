#include "PathSearch.h"
#include <chrono>
#include <algorithm>
#include <iostream>

// Required due to max being redefined by some other import
#undef max

#define DEBUG 2

namespace ufl_cap4053
{
	namespace constants {
		const std::unordered_map<int, std::unordered_set<std::pair<int, int>, searches::pair_hash>> ADJACENT_TILES = { {0, {{1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {0, 1}, {1, 0}}}, {1, {{-1, 0}, {0, -1}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}}} };
		const double HEURISTIC_WEIGHT = 1.0f;
	}

	namespace searches
	{
		PlannerNode::PlannerNode(const Tile* tile, const Tile* parent, double gCost, double hCost) : tile(tile), parent(parent), givenCost(gCost), heuristicCost(hCost) {
				finalCost = givenCost + constants::HEURISTIC_WEIGHT * heuristicCost;
		}

		const std::pair<int, int> PlannerNode::getTileCoords() const {
			return std::make_pair(tile->getRow(), tile->getColumn());
		}

		void PlannerNode::setGivenCost(double newCost) {
			givenCost = newCost;
			finalCost = givenCost + constants::HEURISTIC_WEIGHT * heuristicCost;
		}


		PathSearch::PathSearch() : open(greaterThan) {}

		PathSearch::~PathSearch() {
			shutdown();
			unload();
		}

		//! \brief Called after the tile map is loaded. Generates the search graph
		void PathSearch::load(TileMap* _tileMap) {
			tileMap = _tileMap;
			std::unordered_set<std::pair<int, int>, pair_hash> posNeighbors{};
			std::pair<int, int > neighbor{};
			Tile* keyTile{};
			Tile* neighborTile{};

			int rows = _tileMap->getRowCount();
			int cols = _tileMap->getColumnCount();

			// Iterate over rows
			for (int row{}; row < rows; ++row) {
				// Iterate over cols
				for (int col{}; col < cols; ++col) {
					keyTile = _tileMap->getTile(row, col);
					// Skip if no tile exists at location
					if (!keyTile || (int)keyTile->getWeight() == 0) {
						continue;
					}
					// Initialize tile pair and get relative neighbor indexes
					posNeighbors = constants::ADJACENT_TILES.at(row % 2);
					// Check each posible neighbor
					for (const std::pair<int, int>& move : posNeighbors) {
						// Convert relative movement to row/col in tilemap
						neighbor = std::make_pair(row + move.first, col + move.second);
						// Check if valid position (possibly unnecessary)
						if ((0 <= neighbor.first) && (0 <= neighbor.second) && (neighbor.first < rows) && (neighbor.second < cols)) {
							neighborTile = _tileMap->getTile(neighbor.first, neighbor.second);
							// Add as neighbor in search graph if tile exists
							if (neighborTile && (int)neighborTile->getWeight() != 0) {
								searchGraph[keyTile].insert(neighborTile);
								//If debuging, draw search graph
								if (DEBUG > 2) {
									keyTile->addLineTo(neighborTile, 0xFFFFFFFF);
								}
							}
						}
					}
				}
			}
		}

		//! \brief Called before any update of the path planner,
		//! prepares for search to be performed between the tiles at
		//! the coordinates indicated.This method is always preceded by at least one call to load().
		void PathSearch::initialize(int startRow, int startCol, int goalRow, int goalCol) {
			startTile = tileMap->getTile(startRow, startCol);
			goalTile = tileMap->getTile(goalRow, goalCol);
			isComplete = false;
			PlannerNode* startNode = new PlannerNode(startTile);
			open.push(startNode);
			visited[startTile] = startNode;
			solution.clear();
		}

		//! \brief Runs path planner for the specified timeslice.
		//! If timeslice is 0, will run once. Otherwise, 
		//! will run until time elapses
		void PathSearch::update(long timeslice) {
			if (timeslice == 0) {
				aStarIteration();
				return;
			}

			auto stopPoint = std::chrono::system_clock::now().time_since_epoch() + std::chrono::milliseconds(timeslice);
			auto startIter = std::chrono::system_clock::now().time_since_epoch();
			auto endIter = std::chrono::system_clock::now().time_since_epoch();
			auto maxIterDuration = endIter - startIter;
			while ((endIter + maxIterDuration < stopPoint) && !isComplete) {
				startIter = std::chrono::system_clock::now().time_since_epoch();
				aStarIteration();
				endIter = std::chrono::system_clock::now().time_since_epoch();
				maxIterDuration = max(maxIterDuration, endIter - startIter);
			}
		}

		//! \brief Cleans up memory for this search
		void PathSearch::shutdown() {
			visited.clear();
			open.clear();
		}

		//! \brief Cleans up memory associated with the tile map
		void PathSearch::unload() {
			searchGraph.clear();
		}

		//! \brief Returns true if a solution has been found
		//! and false otherwise.
		bool PathSearch::isDone() const {
			return isComplete;
		}

		std::vector<Tile const*> const PathSearch::getSolution() const {
			return solution;
		}

		// \brief Runs a single iteration of A* search
		void PathSearch::aStarIteration() {
			if (!open.empty()) {
				// Retrieve next node
				PlannerNode* current = open.front();
				open.pop();

				if (DEBUG > 1) {
					tileMap->getTile(current->getTile()->getRow(), current->getTile()->getColumn())->setMarker(0xffff0000);
				}

				//visited[current->getTile()] = current;

				double cost{};
				PlannerNode* next;
				for (const Tile* neighbor : searchGraph.at(current->getTile())) {
					cost = current->getGivenCost() + (int)neighbor->getWeight() * std::sqrt(std::pow(neighbor->getXCoordinate() - current->getTile()->getXCoordinate(), 2) + std::pow(neighbor->getYCoordinate() - current->getTile()->getYCoordinate(), 2));
					// Check if goal
					if (neighbor == goalTile) {
						isComplete = true;
						visited[neighbor] = new PlannerNode(neighbor, current->getTile(), cost, heuristic(neighbor));
						if (DEBUG > 0) { std::cout << "Cost: " << cost << std::endl; }
						buildSolution();
						return;
					}
					
					// Check if visited
					if (visited.find(neighbor) != visited.end()) {
						// Replace node if better path found
						next = visited.at(neighbor);
						if (cost < next->getGivenCost()) {
							open.remove(next);
							next->setGivenCost(cost);
							next->setParent(current->getTile());
							open.push(next);
							if (DEBUG > 0) {
								tileMap->getTile(neighbor->getRow(), neighbor->getColumn())->setMarker(0x7f007f00);
							}
						}
					}
					// If not found
					else {
						next = new PlannerNode(neighbor, current->getTile(), cost, heuristic(neighbor));
						visited[neighbor] = next;
						open.push(next);
						if (DEBUG > 0) {
							tileMap->getTile(neighbor->getRow(), neighbor->getColumn())->setMarker(0x7f007f00);
						}
					}
				}
			}
		}

		void PathSearch::buildSolution() {
			const Tile* current = goalTile;
			while (current) {
				solution.push_back(current);
				current = visited.at(current)->getParent();
			}
		}

		bool PathSearch::areAdjacent(const Tile* lhs, const Tile* rhs) const {
			// Find relative distance
			std::pair<int, int> dist = { lhs->getRow() - rhs->getRow(), lhs->getColumn() - rhs->getColumn() };
			
			return constants::ADJACENT_TILES.at(lhs->getRow() % 2).find(dist) != constants::ADJACENT_TILES.at(lhs->getRow() % 2).end();
		}

		double PathSearch::heuristic(const Tile* src) const {
			return std::sqrt(std::pow(src->getXCoordinate() - goalTile->getXCoordinate(), 2) + std::pow(src->getYCoordinate() - goalTile->getYCoordinate(), 2));
		}
	}
}  // close namespace ufl_cap4053::searches

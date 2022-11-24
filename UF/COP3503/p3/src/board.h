#pragma once
#include <vector>
#include <random>
#include "textureManager.h"
#include "tile.h"
#include "buttonTile.h"
#include "boardConfig.h"
#include "mineCounter.h"

class Board {
	std::vector<Tile> m_tiles;
	int m_rows;
	int m_columns;
	int m_numMines;
	bool m_debugMode;
	bool m_isGameOver;

	ButtonTile m_face;
	ButtonTile m_debug;
	ButtonTile m_test1;
	ButtonTile m_test2;
	ButtonTile m_test3;
	MineCounter m_mineCounter;
	sf::RenderWindow& m_window;

	static std::mt19937 random;
	std::uniform_int_distribution<int> m_dist;

	void initializeTiles();
	void initializeMines();
	int getRandInt() const;
	std::vector<int> getSurroundingTileIndices(const Tile& source);
	int countNeighborsBombs(const std::vector<int>& neighbors) const;
	void revealTile(Tile& tile);
	void boardReset();
	void loadTestConfig(int boardNum);
	void win();
	void lose();
	bool isWinCondition() const;

public:
	Board(sf::RenderWindow& window, BoardConfig config);
	void draw() const;
	void toggleFlag(sf::Vector2i mousePosition);
	void reveal(sf::Vector2i mousePosition);
	void checkButtonSelection(sf::Vector2i mousePosition);
	void clear() { m_mineCounter.clear(); };

	int getNumRows() const { return m_rows; };
	int getNumColumns() const { return m_columns; };
};
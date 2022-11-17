#pragma once
#include <vector>
#include <memory>
#include <string_view>
#include <random>
#include "textureManager.h"
#include "tile.h"
#include "buttonTile.h"
#include "boardConfig.h"

class Board {
	std::vector<Tile> m_tiles;
	size_t m_rows;
	size_t m_columns;
	int m_numMines;
	size_t m_numFlags;
	bool m_debugMode;
	//sf::Sprite m_face;
	//std::vector<sf::Sprite> m_minesDisplay;
	//std::vector<Tile> m_testTiles;

	sf::RenderWindow& m_window;

	static std::mt19937 random;
	std::uniform_int_distribution<int> m_dist;
	void initializeTiles();
	void initializeMines();
	int getRandInt() const;
	std::vector<Tile> getSurroundingTiles(const Tile& source);
	int countNeighborsBombs(const std::vector<Tile>& neighbors) const;

public:
	Board(sf::RenderWindow& window, BoardConfig config);
	//Board(const std::string_view& testBoard, std::shared_ptr<TextureManager> textureManager, std::shared_ptr<sf::RenderWindow> window);
	void draw() const;
	void toggleFlag(sf::Vector2i mousePosition);
	void reveal(sf::Vector2i mousePosition);
};
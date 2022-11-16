#pragma once
#include <vector>
#include <memory>
#include <string_view>
#include <random>
#include <fstream>
#include "textureManager.h"
#include "tile.h"
#include "buttonTile.h"

class Board {
	std::vector<Tile> tiles;
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

public:
	Board(sf::RenderWindow& window);
	//Board(const std::string_view& testBoard, std::shared_ptr<TextureManager> textureManager, std::shared_ptr<sf::RenderWindow> window);
	void draw() const;
};
#pragma once
#include <vector>
#include <memory>
#include <string_view>
#include "textureManager.h"
#include "tile.h"

class Board {
	std::vector<Tile> tiles;
	size_t rows;
	size_t columns;
	size_t numMines;
	size_t numFlags;
	bool debugMode;
	sf::Sprite face;
	std::vector<sf::Sprite> minesDisplay;
	std::vector<Tile> testTiles;

	std::shared_ptr<TextureManager> m_textureManager;
	std::shared_ptr<sf::RenderWindow> m_window;
};
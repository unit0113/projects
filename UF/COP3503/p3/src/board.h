#pragma once
#include <vector>
#include <memory>
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

	sf::RenderWindow window;
	std::shared_ptr<TextureManager> textureManager;
};
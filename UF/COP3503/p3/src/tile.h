#pragma once
#include <SFML\Graphics.hpp>
#include <string>
#include <memory>
#include "textureManager.h"

class Tile {
	sf::Sprite m_background;
	sf::Sprite m_image;
	sf::Vector2i m_location;
	bool m_isBomb;
	bool m_isRevealed;
	std::shared_ptr<> m_textureManager;

	std::string countAdjacentBombs();

public:
	Tile(size_t x, size_t y);
	void draw();
	void reveal();
};



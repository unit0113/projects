#pragma once
#include <SFML\Graphics.hpp>
#include <unordered_map>
#include <string>

class Tile {
	sf::Sprite m_background;
	sf::Sprite m_image;
	sf::Vector2i m_location;
	bool m_isBomb;

	std::string countAdjacentBombs();

	static std::unordered_map<std::string, sf::Sprite> initSpriteMap();
	const static std::unordered_map<std::string, sf::Sprite> spriteMap;

public:
	Tile(size_t x, size_t y);
	void draw();
	void reveal();
};



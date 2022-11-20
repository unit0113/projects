#pragma once
#include <SFML\Graphics.hpp>
#include <unordered_map>
#include <vector>
#include <string>

class MineCounter {
	int m_numMines;
	sf::RenderWindow& m_window;
	std::vector<sf::Sprite> m_display;
	std::unordered_map<std::string, sf::Texture> textureMap;

public:
	MineCounter(sf::RenderWindow& window, int numMines, int height);
	void draw();
	void clear();

	MineCounter& operator++();
	MineCounter& operator--();
};
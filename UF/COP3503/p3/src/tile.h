#pragma once
#include <SFML\Graphics.hpp>
#include <string>
#include <memory>
#include "textureManager.h"

class Tile {
	sf::Sprite m_background;
	sf::Sprite m_image;
	bool m_isBomb;
	bool m_isRevealed;
	bool m_isFlag;
	sf::RenderWindow& m_window;

public:
	Tile(float x, float y, sf::RenderWindow& window);
	void draw() const;
	void setAsBomb();
	void toggleFlag();
	//bool reveal();

	bool isBomb() const { return m_isBomb; };
	bool isRevealed() const { return m_isRevealed; };
	bool isFlag() const { return m_isFlag; };

	bool contains(float x, float y) const { return m_background.getGlobalBounds().contains(x, y); };
};



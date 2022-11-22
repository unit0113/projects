#pragma once
#include <SFML\Graphics.hpp>
#include <string>
#include <memory>
#include "textureManager.h"

class Tile {
	sf::Sprite m_background;
	sf::Sprite m_mineImage;
	sf::Sprite m_flagImage;
	sf::Sprite m_mineCountImage;
	bool m_isBomb;
	bool m_isRevealed;
	bool m_isFlag;
	sf::RenderWindow& m_window;

	void setImageLocations(float x, float y);

public:
	Tile(float x, float y, sf::RenderWindow& window);
	Tile(float x, float y, sf::RenderWindow& window, bool isBomb);
	void draw() const;
	void drawDebug() const;
	void setAsBomb();
	void toggleFlag();
	void reveal(int bombs);

	bool isBomb() const { return m_isBomb; };
	bool isRevealed() const { return m_isRevealed; };
	bool isFlag() const { return m_isFlag; };
	bool isRevealable() const { return (!isRevealed() && !isFlag()); };

	bool contains(sf::Vector2i position) const { return m_background.getGlobalBounds().contains(position.x, position.y); };

	bool operator==(const Tile& other) const;
};



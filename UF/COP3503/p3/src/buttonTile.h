#pragma once
#include <SFML\Graphics.hpp>
#include <string>
#include <memory>
#include "textureManager.h"

class ButtonTile {
	sf::Sprite m_sprite;
	sf::RenderWindow& m_window;

public:
	ButtonTile(float x, float y, sf::RenderWindow& window, const std::string texture);
	void setTexture(std::string newTexture);
	void draw() const;
	bool contains(float x, float y) const { return m_sprite.getGlobalBounds().contains(x, y); };
};
#pragma once
#include <SFML\Graphics.hpp>
#include <string_view>
#include <memory>
#include "textureManager.h"

class Tile {
	sf::Sprite m_background;
	sf::Sprite m_image;
	bool m_isBomb;
	bool m_isRevealed;
	bool m_isFlag;
	std::shared_ptr<TextureManager> m_textureManager;
	std::shared_ptr<sf::RenderWindow> m_window;

public:
	Tile(float x, float y, std::shared_ptr<TextureManager> textureManager, std::shared_ptr<sf::RenderWindow> window);
	void draw() const;
	void setAsBomb();
	void flag();
	bool reveal(const std::string_view& contents);

	bool isBomb() const { return m_isBomb; };
	bool isRevealed() const { return m_isRevealed; };
	bool isFlag() const { return m_isFlag; };

	bool contains(float x, float y) const { return m_background.getGlobalBounds().contains(x, y); };
};



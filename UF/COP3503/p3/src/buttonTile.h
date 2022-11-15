#pragma once
#include <SFML\Graphics.hpp>
#include <string_view>
#include <memory>
#include "textureManager.h"

class ButtonTile {
	sf::Sprite m_sprite;
	std::shared_ptr<TextureManager> m_textureManager;
	std::shared_ptr<sf::RenderWindow> m_window;

public:
	ButtonTile(float x, float y, std::shared_ptr<TextureManager> textureManager, std::shared_ptr<sf::RenderWindow> window, const std::string_view& sprite);
	void setSprite(std::string_view newSprite);
	void draw() const;
	bool contains(float x, float y) const { return m_sprite.getGlobalBounds().contains(x, y); };
};
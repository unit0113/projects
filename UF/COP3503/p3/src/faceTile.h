#pragma once
#include <SFML\Graphics.hpp>
#include <string_view>
#include <memory>
#include "textureManager.h"

class FaceTile {
	sf::Sprite m_sprite;
	std::shared_ptr<TextureManager> m_textureManager;
	std::shared_ptr<sf::RenderWindow> m_window;

public:
	FaceTile(std::shared_ptr<TextureManager> textureManager, std::shared_ptr<sf::RenderWindow> window);
	void setSprite(std::string_view newFace);
	void draw() const;
};
#pragma once
#include <SFML\Graphics.hpp>
#include <string_view>
#include <memory>
#include "textureManager.h"

class DebugTile {
	sf::Sprite m_sprite;
	std::shared_ptr<TextureManager> m_textureManager;
	std::shared_ptr<sf::RenderWindow> m_window;
	bool m_debugModeActive;

public:
	DebugTile(std::shared_ptr<TextureManager> textureManager, std::shared_ptr<sf::RenderWindow> window);
	void draw() const;
	bool toggle();
};
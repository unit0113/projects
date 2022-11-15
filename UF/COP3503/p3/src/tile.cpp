#include "tile.h"

Tile::Tile(float x, float y, std::shared_ptr<TextureManager> textureManager, std::shared_ptr<sf::RenderWindow> window)
	: m_textureManager(textureManager), m_background(m_textureManager->getTexture("hidden")), m_window(window){
	m_isBomb = false;
	m_isRevealed = false;
	m_isFlag = false;
	m_background.setPosition(x, y);
	m_image.setPosition(x, y);
}

void Tile::draw() const {
	m_window->draw(m_background);
	if (m_image.getTexture()) {
		m_window->draw(m_image);
	}
}

void Tile::setAsBomb() {
	m_isBomb = true;
}

void Tile::flag() {
	m_isFlag = true;
	m_image.setTexture(m_textureManager->getTexture("flag"));
}

// Return true if trigger recursive reveal of surrounding tiles
bool Tile::reveal(const std::string_view& contents) {
	m_background.setTexture(m_textureManager->getTexture("revealed"));
	if (contents != "") {
		m_image.setTexture(m_textureManager->getTexture(contents));
		return false;
	}
	else {
		return true;
	}
}
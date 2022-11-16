#include "tile.h"

Tile::Tile(float x, float y, sf::RenderWindow& window)
	: m_background(TextureManager::getTexture("hidden")), m_window(window){
	m_isBomb = false;
	m_isRevealed = false;
	m_isFlag = false;
	m_background.setPosition(x, y);
	m_image.setPosition(x, y);
}

void Tile::draw() const {
	m_window.draw(m_background);
	if (m_image.getTexture()) {
		m_window.draw(m_image);
	}
}

void Tile::setAsBomb() {
	m_isBomb = true;
}

void Tile::toggleFlag() {
	m_isFlag = true;
	m_image.setTexture(TextureManager::getTexture("flag"));
}

// Return true if trigger recursive reveal of surrounding tiles
/*bool Tile::reveal() {
	m_background.setTexture(m_textureManager.getTexture("revealed"));
	if (contents != "") {
		m_image.setTexture(m_textureManager.getTexture(contents));
		return false;
	}
	else {
		return true;
	}
}*/
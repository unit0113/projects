#include "tile.h"

Tile::Tile(float x, float y, sf::RenderWindow& window)
	: m_background(TextureManager::getTexture("tile_hidden")), m_window(window){
	m_isBomb = false;
	m_isRevealed = false;
	m_isFlag = false;
	m_background.setPosition(x, y);
	m_image.setPosition(x, y);
}

void Tile::draw() const {
	m_window.draw(m_background);
	if ((m_isRevealed || m_isFlag) && m_image.getTexture()) {
		m_window.draw(m_image);
	}
}

void Tile::setAsBomb() {
	m_isBomb = true;
	m_image.setTexture(TextureManager::getTexture("mine"));
}

void Tile::toggleFlag() {
	m_isFlag = !m_isFlag;
	if (m_isFlag) {
		m_image.setTexture(TextureManager::getTexture("flag"));
	}
	else {
		m_image = sf::Sprite();
		m_image.setPosition(m_background.getPosition());
	}
	
}

void Tile::reveal(int bombs) {
	m_isRevealed = true;
	m_background.setTexture(TextureManager::getTexture("tile_revealed"));
	if (!m_isBomb && bombs != 0) {
		m_image.setTexture(TextureManager::getTexture("number_" + std::to_string(bombs)));
	}
}

#include "tile.h"

Tile::Tile(float x, float y, sf::RenderWindow& window)
	: m_background(TextureManager::getTexture("tile_hidden")), m_window(window),
	m_mineImage(TextureManager::getTexture("mine")), m_flagImage(TextureManager::getTexture("flag")) {

	m_isBomb = false;
	m_isRevealed = false;
	m_isFlag = false;
	setImageLocations(x, y);
}

Tile::Tile(float x, float y, sf::RenderWindow& window, bool isBomb)
	: m_background(TextureManager::getTexture("tile_hidden")), m_window(window), m_isBomb(isBomb),
	m_mineImage(TextureManager::getTexture("mine")), m_flagImage(TextureManager::getTexture("flag")) {

	m_isRevealed = false;
	m_isFlag = false;
	setImageLocations(x, y);
}

void Tile::setImageLocations(float x, float y) {
	m_background.setPosition(x, y);
	m_mineImage.setPosition(x, y);
	m_flagImage.setPosition(x, y);
	m_mineCountImage.setPosition(x, y);
	}

void Tile::draw() const {
	m_window.draw(m_background);
	if (m_mineCountImage.getTexture()) m_window.draw(m_mineCountImage);
	if (m_isFlag) m_window.draw(m_flagImage);
	else if (m_isBomb && m_isRevealed) m_window.draw(m_mineImage);
}

void Tile::drawDebug() const {
	m_window.draw(m_background);
	if (m_mineCountImage.getTexture()) m_window.draw(m_mineCountImage);
	if (m_isBomb) m_window.draw(m_mineImage);
	else if (m_isFlag) m_window.draw(m_flagImage);
}

void Tile::setAsBomb() {
	m_isBomb = true;
}

void Tile::toggleFlag() {
	m_isFlag = !m_isFlag;
}

void Tile::reveal(int bombs) {
	m_isRevealed = true;
	m_background.setTexture(TextureManager::getTexture("tile_revealed"));
	if (!m_isBomb && bombs != 0) {
		m_mineCountImage.setTexture(TextureManager::getTexture("number_" + std::to_string(bombs)));
	}
}

bool Tile::operator==(const Tile& other) const {
	return (m_background.getPosition() == other.m_background.getPosition());
}

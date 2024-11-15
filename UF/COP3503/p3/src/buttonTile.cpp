#include "buttonTile.h"

ButtonTile::ButtonTile(float x, float y, sf::RenderWindow& window, const std::string texture)
	: m_sprite(TextureManager::getTexture(texture)), m_window(window) {
	m_sprite.setPosition(x, y);
}

void ButtonTile::setTexture(std::string newTexture) {
	m_sprite.setTexture(TextureManager::getTexture(newTexture));
}

void ButtonTile::draw() const {
	m_window.draw(m_sprite);
}

void ButtonTile::swapTexture(std::string texture) {
	m_sprite.setTexture(TextureManager::getTexture(texture));
}

void ButtonTile::reposition(int x, int y) {
	m_sprite.setPosition(x, y);
}

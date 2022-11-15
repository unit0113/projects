#include "buttonTile.h"

ButtonTile::ButtonTile(float x, float y, std::shared_ptr<TextureManager> textureManager, std::shared_ptr<sf::RenderWindow> window, const std::string_view& sprite)
	: m_textureManager(textureManager), m_sprite(m_textureManager->getTexture(sprite)), m_window(window) {
	m_sprite.setPosition(x, y);
}

void ButtonTile::setSprite(std::string_view newSprite) {
	m_sprite.setTexture(m_textureManager->getTexture(newSprite));
}

void ButtonTile::draw() const {
	m_window->draw(m_sprite);
}
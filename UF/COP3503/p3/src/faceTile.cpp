#include "faceTile.h"

FaceTile::FaceTile(std::shared_ptr<TextureManager> textureManager, std::shared_ptr<sf::RenderWindow> window)
	: m_textureManager(textureManager), m_sprite(m_textureManager->getTexture("happy")), m_window(window) {}

void FaceTile::setSprite(std::string_view newFace) {
	m_sprite.setTexture(m_textureManager->getTexture(newFace));
}

void FaceTile::draw() const {
	m_window->draw(m_sprite);
}
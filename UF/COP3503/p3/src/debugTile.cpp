#include "debugTile.h"

DebugTile::DebugTile(std::shared_ptr<TextureManager> textureManager, std::shared_ptr<sf::RenderWindow> window)
	: m_textureManager(textureManager), m_sprite(m_textureManager->getTexture("debug")), m_window(window) {
	m_debugModeActive = false;
}

void DebugTile::draw() const {
	m_window->draw(m_sprite);
}

bool DebugTile::toggle() {
	m_debugModeActive = !m_debugModeActive;
	return m_debugModeActive;
}
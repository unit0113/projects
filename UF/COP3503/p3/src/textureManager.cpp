#include "textureManager.h"

std::unordered_map<std::string, sf::Texture> TextureManager::textureMap;

sf::Texture& TextureManager::getTexture(const std::string texture) {
	if (textureMap.find(texture) == textureMap.end()) {
		textureMap[texture].loadFromFile("images/" + texture + ".png");
	}
	return textureMap[texture];
}

void TextureManager::clear() {
	textureMap.clear();
}

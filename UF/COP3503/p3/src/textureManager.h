#pragma once
#include <SFML\Graphics.hpp>
#include <unordered_map>
#include <string_view>

class TextureManager {
	std::unordered_map<std::string_view, sf::Texture> textureMap;

public:
	TextureManager();
	sf::Texture getTexture(const std::string_view& texture) {return textureMap[texture];};
};
#pragma once
#include <SFML\Graphics.hpp>
#include <unordered_map>
#include <string>

class TextureManager {
	static std::unordered_map<std::string, sf::Texture> textureMap;

public:
	static sf::Texture& getTexture(const std::string texture);
	static void clear();
};
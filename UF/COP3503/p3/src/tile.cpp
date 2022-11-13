#include "tile.h"

std::unordered_map<std::string, sf::Sprite> Tile::initSpriteMap() {
	std::unordered_map<std::string, sf::Sprite> result;
	sf::Texture texture;
	sf::Sprite sprite;
	texture.loadFromFile("images/debug.png");
	sprite.setTexture(texture);
	result["debug"] = sprite;

	texture.loadFromFile("images/digits.png");
	sprite.setTexture(texture);
	result["digits"] = sprite;

	texture.loadFromFile("images/face_happy.png");
	sprite.setTexture(texture);
	result["happy"] = sprite;

	texture.loadFromFile("images/face_lose.png");
	sprite.setTexture(texture);
	result["lose"] = sprite;

	texture.loadFromFile("images/face_win.png");
	sprite.setTexture(texture);
	result["win"] = sprite;

	texture.loadFromFile("images/flag.png");
	sprite.setTexture(texture);
	result["flag"] = sprite;

	texture.loadFromFile("images/mine.png");
	sprite.setTexture(texture);
	result["mine"] = sprite;

	texture.loadFromFile("images/number_1.png");
	sprite.setTexture(texture);
	result["1"] = sprite;

	texture.loadFromFile("images/number_2.png");
	sprite.setTexture(texture);
	result["2"] = sprite;

	texture.loadFromFile("images/number_3.png");
	sprite.setTexture(texture);
	result["3"] = sprite;

	texture.loadFromFile("images/number_4.png");
	sprite.setTexture(texture);
	result["4"] = sprite;

	texture.loadFromFile("images/number_5.png");
	sprite.setTexture(texture);
	result["5"] = sprite;

	texture.loadFromFile("images/number_6.png");
	sprite.setTexture(texture);
	result["6"] = sprite;

	texture.loadFromFile("images/number_7.png");
	sprite.setTexture(texture);
	result["7"] = sprite;

	texture.loadFromFile("images/number_8.png");
	sprite.setTexture(texture);
	result["8"] = sprite;

	texture.loadFromFile("images/test_1.png");
	sprite.setTexture(texture);
	result["test_1"] = sprite;

	texture.loadFromFile("images/test_2.png");
	sprite.setTexture(texture);
	result["test_2"] = sprite;

	texture.loadFromFile("images/test_3.png");
	sprite.setTexture(texture);
	result["test_3"] = sprite;

	texture.loadFromFile("images/tile_hidden.png");
	sprite.setTexture(texture);
	result["hidden"] = sprite;

	texture.loadFromFile("tile_revealed.png");
	sprite.setTexture(texture);
	result["revealed"] = sprite;

	return result;
}

const std::unordered_map<std::string, sf::Sprite> Tile::spriteMap = Tile::initSpriteMap();

Tile::Tile(size_t x, size_t y) : m_location(x, y), m_background(Tile::spriteMap["hidden"]) {
	m_isBomb = false;
}
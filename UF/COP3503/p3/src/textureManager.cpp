#include "textureManager.h"

TextureManager::TextureManager() {
	sf::Texture texture;

	texture.loadFromFile("images/debug.png");
	textureMap["debug"] = texture;

	// Load seperate Digits: https://www.sfml-dev.org/tutorials/2.5/graphics-sprite.php
	size_t i;
	for (i = 0; i <= 9; ++i) {
		texture.loadFromFile("images/digits.png", sf::IntRect(21 * i, 0, 21, 32));
		textureMap[std::to_string(i)] = texture;
	}
	texture.loadFromFile("images/digits.png", sf::IntRect(21 * i, 0, 21, 32));
	textureMap["-"] = texture;

	texture.loadFromFile("images/face_happy.png");
	textureMap["happy"] = texture;

	texture.loadFromFile("images/face_lose.png");
	textureMap["lose"] = texture;

	texture.loadFromFile("images/face_win.png");
	textureMap["win"] = texture;

	texture.loadFromFile("images/flag.png");
	textureMap["flag"] = texture;

	texture.loadFromFile("images/mine.png");
	textureMap["mine"] = texture;

	texture.loadFromFile("images/number_1.png");
	textureMap["1"] = texture;

	texture.loadFromFile("images/number_2.png");
	textureMap["2"] = texture;

	texture.loadFromFile("images/number_3.png");
	textureMap["3"] = texture;

	texture.loadFromFile("images/number_4.png");
	textureMap["4"] = texture;

	texture.loadFromFile("images/number_5.png");
	textureMap["5"] = texture;

	texture.loadFromFile("images/number_6.png");
	textureMap["6"] = texture;

	texture.loadFromFile("images/number_7.png");
	textureMap["7"] = texture;

	texture.loadFromFile("images/number_8.png");
	textureMap["8"] = texture;

	texture.loadFromFile("images/test_1.png");
	textureMap["test_1"] = texture;

	texture.loadFromFile("images/test_2.png");
	textureMap["test_2"] = texture;

	texture.loadFromFile("images/test_3.png");
	textureMap["test_3"] = texture;

	texture.loadFromFile("images/tile_hidden.png");
	textureMap["hidden"] = texture;

	texture.loadFromFile("tile_revealed.png");
	textureMap["revealed"] = texture;
}

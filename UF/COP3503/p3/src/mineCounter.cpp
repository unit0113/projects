#include "mineCounter.h"
#include <cmath>

MineCounter::MineCounter(sf::RenderWindow& window, int numMines, int height) : m_window(window), m_numMines(numMines) {
	//https://www.sfml-dev.org/tutorials/2.5/graphics-sprite.php
	size_t i;
	for (i = 0; i <= 9; ++i) {
		textureMap[std::to_string(i)].loadFromFile("images/digits.png", sf::IntRect(i * 21, 0, 21, 32));
	}
	textureMap["-"].loadFromFile("images/digits.png", sf::IntRect(i * 21, 0, 21, 32));

	for (int i = 3; i >= 0; --i) {
		sf::Sprite sprite;
		sprite.setPosition(i * 21, height);
		m_display.push_back(sprite);
	}
	setTextures();
	m_display[3].setTexture(textureMap["-"]);
}

void MineCounter::draw() const {
	m_window.draw(m_display[0]);
	m_window.draw(m_display[1]);
	m_window.draw(m_display[2]);

	if (m_numMines < 0) m_window.draw(m_display[3]);
}

void MineCounter::clear() {
	textureMap.clear();
}

void MineCounter::setTextures() {
	int num = m_numMines;
	for (size_t i{}; i < 3; ++i) {
		m_display[i].setTexture(textureMap[std::to_string(std::abs(num % 10))]);
		m_window.draw(m_display[i]);
		num /= 10;
	}
}

void MineCounter::reset(int numMines, int height) {
	m_numMines = numMines;
	int i{ 3 };
	for (sf::Sprite& sprite : m_display) {
		sprite.setPosition(i * 21, height);
		--i;
	}
	setTextures();
}

void MineCounter::setWin() {
	m_numMines = 0;
	setTextures();
}

MineCounter& MineCounter::operator++() {
	++m_numMines;
	setTextures();
	return *this;
}

MineCounter& MineCounter::operator--() {
	--m_numMines;
	setTextures();
	return *this;
}

#include <ctime>
#include <stdexcept>
#include "board.h"

std::mt19937 Board::random(time(0));


Board::Board(sf::RenderWindow& window)
	: m_dist(0, m_numMines), m_debugMode(false), m_window(window) {
	//Read config
	std::ifstream cfgFile("boards/config.cfg");
	if (!cfgFile) throw std::runtime_error("Error: File not Found");
	cfgFile >> m_columns;
	cfgFile >> m_rows;
	cfgFile >> m_numMines;
	

	initializeTiles();
	initializeMines();
}

void Board::initializeTiles() {
	size_t x{};
	size_t y{};
	for (size_t col{}; col < m_columns; ++col) {
		for (size_t row{}; row < m_rows; row++) {
			tiles.push_back(Tile(x, y, m_window));
			x += 32;
		}
		x = 0;
		y += 32;
	}
}

void Board::initializeMines() {
	int tileIndex;
	for (size_t i{}; i < m_numMines; ++i) {
		do {
			tileIndex = m_dist(random);
		} while (tiles[tileIndex].isBomb());
		tiles[tileIndex].setAsBomb();
	}
}

int Board::getRandInt() const{
	std::uniform_int_distribution<int> dist(0, m_numMines);
	return dist(random);
}

void Board::draw() const {
	for (Tile t : tiles) {
		t.draw();
	}

	//m_face.draw();
	//m_minesDisplay.draw();
	/*for (Tile testT: m)testTiles) {
		testT.draw();
	}*/
}
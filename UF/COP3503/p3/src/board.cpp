#include <ctime>
#include "board.h"
#include <iostream>

std::mt19937 Board::random(time(0));

Board::Board(sf::RenderWindow& window, BoardConfig config)
	: m_dist(0, config.m_columns * config.m_rows - 1),
	m_debugMode(false), m_window(window),
	m_face((config.m_columns -1) * 16, config.m_rows * 32, window, "face_happy"),
	m_test3((config.m_columns - 2) * 32, config.m_rows * 32, window, "test_3"),
	m_test2((config.m_columns - 4) * 32, config.m_rows * 32, window, "test_2"),
	m_test1((config.m_columns - 6) * 32, config.m_rows * 32, window, "test_1"),
	m_debug((config.m_columns - 8) * 32, config.m_rows * 32, window, "debug"),
	m_mineCounter(window, config.m_numMines, config.m_rows * 32) {

	m_columns = config.m_columns;
	m_rows = config.m_rows;
	m_numMines = config.m_numMines;

	initializeTiles();
	initializeMines();
}

void Board::initializeTiles() {
	size_t x{};
	size_t y{};
	for (size_t row{}; row < m_rows; ++row) {
		for (size_t col{}; col < m_columns; col++) {
			m_tiles.push_back(Tile(x, y, m_window));
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
		} while (m_tiles[tileIndex].isBomb());
		m_tiles[tileIndex].setAsBomb();
	}
}

int Board::getRandInt() const{
	std::uniform_int_distribution<int> dist(0, m_numMines);
	return dist(random);
}

std::vector<int> Board::getSurroundingTileIndices(const Tile& source) {
	//https://www.delftstack.com/howto/cpp/find-in-vector-in-cpp/
	std::vector<Tile>::iterator it = std::find(m_tiles.begin(), m_tiles.end(), source);
	int index = std::distance(m_tiles.begin(), it);

	std::vector<int> neighbors;
	// https://stackoverflow.com/questions/9355537/finding-neighbors-of-2d-array-when-represented-as-1d-array
	// North
	if ((index - m_columns) >= 0) neighbors.push_back(index - m_columns);
	// South
	if ((index + m_columns) < m_tiles.size()) neighbors.push_back(index + m_columns);
	// East
	if (((index + 1) % m_columns) != 0) neighbors.push_back(index + 1);
	// West
	if ((index % m_columns) != 0) neighbors.push_back(index - 1);
	// NE
	if ((index - m_columns + 1) >= 0 && ((index + 1) % m_columns) != 0) neighbors.push_back(index - m_columns + 1);
	// NW
	if ((index - m_columns - 1) >= 0 && (index % m_columns) != 0) neighbors.push_back(index - m_columns - 1);
	// SE
	if ((index + m_columns + 1) < m_tiles.size() && ((index + 1) % m_columns) != 0) neighbors.push_back(index + m_columns + 1);
	// SW
	if ((index + m_columns - 1) < m_tiles.size() && (index % m_columns) != 0) neighbors.push_back(index + m_columns - 1);

	return neighbors;
}

int Board::countNeighborsBombs(const std::vector<int>& neighbors) const {
	int bombs{};
	for (const int& index : neighbors) {
		if (m_tiles[index].isBomb()) ++bombs;
	}
	return bombs;
}

void Board::draw() {
	for (const Tile& t : m_tiles) {
		t.draw();
	}

	m_face.draw();
	m_debug.draw();
	m_test1.draw();
	m_test2.draw();
	m_test3.draw();

	m_mineCounter.draw();
}

void Board::toggleFlag(sf::Vector2i mousePosition) {
	for (Tile& t : m_tiles) {
		if (t.contains(mousePosition)) {
			if (!t.isRevealed()) {
				if (t.isFlag()) {
					++m_mineCounter;
				}
				else {
					--m_mineCounter;
				}
				t.toggleFlag();
			}
			break;
		}
	}
}

void Board::reveal(sf::Vector2i mousePosition) {
	for (Tile& t : m_tiles) {
		if (t.contains(mousePosition)) {
			revealTile(t);
			break;
		}
	}
}

void Board::revealTile(Tile& tile) {
	if (tile.isRevealable()) {
		std::vector<int> neighborIndices = getSurroundingTileIndices(tile);
		size_t surroundingNumBombs = countNeighborsBombs(neighborIndices);
		tile.reveal(surroundingNumBombs);
		if (!tile.isBomb() && surroundingNumBombs == 0) {
			for (const size_t& index : neighborIndices) {
				revealTile(m_tiles[index]);
			}
		}
	}
}

void Board::boardReset() {
	BoardConfig config;
	m_dist = std::uniform_int_distribution<int>(0, config.m_columns * config.m_rows - 1);
	m_debugMode = false;
	m_columns = config.m_columns;
	m_rows = config.m_rows;
	m_numMines = config.m_numMines;
	m_face.reposition((m_columns - 1) * 16, m_rows * 32);
	m_test3.reposition((m_columns - 2) * 32, m_rows * 32);
	m_test2.reposition((m_columns - 4) * 32, m_rows * 32);
	m_test1.reposition((m_columns - 6) * 32, m_rows * 32);
	m_debug.reposition((m_columns - 8) * 32, m_rows * 32);

	m_tiles.clear();
	initializeTiles();
	initializeMines();

	m_window.setSize(sf::Vector2u(config.m_columns * 32, config.m_rows * 32 + 100));
}

void Board::checkButtonSelection(sf::Vector2i mousePosition) {
	if (m_face.contains(mousePosition)) {
		boardReset();
	}
}

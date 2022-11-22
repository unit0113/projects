#include <ctime>
#include "board.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

std::mt19937 Board::random(time(0));

Board::Board(sf::RenderWindow& window, BoardConfig config)
	: m_dist(0, config.m_columns * config.m_rows - 1),
	m_debugMode(false),
	m_isDead(false),
	m_window(window),
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
	// https://www.delftstack.com/howto/cpp/find-in-vector-in-cpp/
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

void Board::draw() const {
	if (m_debugMode) {
		for (const Tile& t : m_tiles) {
			t.drawDebug();
		}
	}
	else {
		for (const Tile& t : m_tiles) {
			t.draw();
		}
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
	if (isWinCondition()) win();		// Game win
}

void Board::reveal(sf::Vector2i mousePosition) {
	if (m_isDead) return;				// Prevent reveals if game over

	for (Tile& t : m_tiles) {
		if (t.contains(mousePosition)) {
			revealTile(t);
			if (t.isBomb()) lose();		// Game over
			break;
		}
	}
	if (isWinCondition()) win();		// Game win
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
	m_debugMode = false;
	m_isDead = false;
	BoardConfig config;
	m_numMines = config.m_numMines;

	m_tiles.clear();
	initializeTiles();
	initializeMines();

	m_mineCounter.reset(m_numMines, m_rows * 32);
	m_face.setTexture("face_happy");
}

void Board::checkButtonSelection(sf::Vector2i mousePosition) {
	if (m_face.contains(mousePosition)) {
		boardReset();
	}
	else if (m_debug.contains(mousePosition)) {
		m_debugMode = !m_debugMode;
	}
	else if (m_test1.contains(mousePosition)) {
		loadTestConfig(1);
	}
	else if (m_test2.contains(mousePosition)) {
		loadTestConfig(2);
	}
	else if (m_test3.contains(mousePosition)) {
		loadTestConfig(3);
	}
}

void Board::loadTestConfig(int boardNum) {
	m_tiles.clear();

	std::istringstream inStream;
	std::string inLine;
	char isMine{};
	int rowCount{};
	int bombCount{};
	int x{};
	int y{};
	std::ifstream file("boards/testboard" + std::to_string(boardNum) + ".brd");

	while (std::getline(file, inLine)) {
		inStream.clear();
		inStream.str(inLine);
		while (inStream >> isMine) {
			m_tiles.push_back(Tile(x, y, m_window, isMine == '1'));
			x += 32;

			if (isMine == '1') ++bombCount;
		}
		x = 0;
		y += 32;	
		++rowCount;
	}

	file.close();
	file.seekg(0, std::ios::beg);
	std::getline(file, inLine);

	m_columns = inLine.length();
	m_rows = rowCount;
	m_numMines = bombCount;
	m_debugMode = false;

	m_face.reposition((m_columns - 1) * 16, m_rows * 32);
	m_test3.reposition((m_columns - 2) * 32, m_rows * 32);
	m_test2.reposition((m_columns - 4) * 32, m_rows * 32);
	m_test1.reposition((m_columns - 6) * 32, m_rows * 32);
	m_debug.reposition((m_columns - 8) * 32, m_rows * 32);
	m_mineCounter.reset(m_numMines, m_rows * 32);
	m_window.setSize(sf::Vector2u(m_columns * 32, m_rows * 32 + 100));

}

void Board::win() {
	for (Tile& t : m_tiles) {
		if (t.isBomb() and !t.isFlag()) t.toggleFlag();
	}
	m_mineCounter.setWin();
	m_face.setTexture("face_win");
}

void Board::lose() {
	m_isDead = true;
	m_debugMode = true;
	m_face.setTexture("face_lose");
}

bool Board::isWinCondition() const {
	return std::all_of(m_tiles.begin(), m_tiles.end(), [](Tile t) { return t.isFlag() && t.isBomb() || t.isRevealed() || t.isBomb(); });
}

#include <ctime>
#include "board.h"

std::mt19937 Board::random(time(0));

Board::Board(sf::RenderWindow& window, BoardConfig config)
	: m_dist(0, config.m_columns * config.m_rows - 1), m_debugMode(false), m_window(window) {
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

std::vector<Tile> Board::getSurroundingTiles(const Tile& source) {
	//https://www.delftstack.com/howto/cpp/find-in-vector-in-cpp/
	std::vector<Tile>::iterator it = std::find(m_tiles.begin(), m_tiles.end(), source);
	int index = std::distance(m_tiles.begin(), it);

	std::vector<Tile> neighbors;
	// https://stackoverflow.com/questions/9355537/finding-neighbors-of-2d-array-when-represented-as-1d-array
	// North
	if ((index - m_columns) >= 0) neighbors.push_back(m_tiles[index - m_columns]);
	// South
	if ((index + m_columns) < m_tiles.size()) neighbors.push_back(m_tiles[index + m_columns]);
	// East
	if (((index + 1) % m_columns) != 0) neighbors.push_back(m_tiles[index + 1]);
	// West
	if ((index % m_columns) != 0) neighbors.push_back(m_tiles[index - 1]);
	// NE
	if ((index - m_columns + 1) >= 0 && ((index + 1) % m_columns) != 0) neighbors.push_back(m_tiles[index - m_columns + 1]);
	// NW
	if ((index - m_columns - 1) >= 0 && (index % m_columns) != 0) neighbors.push_back(m_tiles[index - m_columns - 1]);
	// SE
	if ((index + m_columns + 1) < m_tiles.size() && ((index + 1) % m_columns) != 0) neighbors.push_back(m_tiles[index + m_columns + 1]);
	// SW
	if ((index + m_columns - 1) < m_tiles.size() && (index % m_columns) != 0) neighbors.push_back(m_tiles[index + m_columns - 1]);

	return neighbors;
}

int Board::countNeighborsBombs(const std::vector<Tile>& neighbors) const {
	int bombs{};
	for (Tile t : neighbors) {
		if (t.isBomb()) ++bombs;
	}
	return bombs;
}

void Board::draw() const {
	for (Tile t : m_tiles) {
		t.draw();
	}

	//m_face.draw();
	//m_minesDisplay.draw();
	/*for (Tile testT: m)testTiles) {
		testT.draw();
	}*/
}

void Board::toggleFlag(sf::Vector2i mousePosition) {
	for (Tile& t : m_tiles) {
		if (t.contains(mousePosition)) {
			if (!t.isRevealed()) {
				t.toggleFlag();
			}
			break;
		}
	}
}

void Board::reveal(sf::Vector2i mousePosition) {
	for (Tile& t : m_tiles) {
		if (t.contains(mousePosition)) {
			if (!t.isRevealed() and !t.isFlag()) {
				std::vector<Tile> neighbors = getSurroundingTiles(t);
				t.reveal(countNeighborsBombs(neighbors));
				// if (countNeighborsBombs(neighbors) == 0) recursive reveal
			}
			break;
		}
	}
}

bool Tile::operator==(const Tile& other) const {
	return (m_background.getPosition() == other.m_background.getPosition());
}
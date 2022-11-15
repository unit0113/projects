#include <ctime>
#include "board.h"

Board::Board() : m_dist(0, m_numMines) {
	//Read config



	initializeMines();
}




std::mt19937 Board::random(time(0));

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
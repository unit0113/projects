#include "tile.h"

Tile::Tile(size_t x, size_t y) : m_location(x, y), m_background(Tile::spriteMap["hidden"]) {
	m_isBomb = false;
}
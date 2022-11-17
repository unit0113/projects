#pragma once
#include <fstream>
#include <stdexcept>

struct BoardConfig {
    size_t m_rows;
    size_t m_columns;
    int m_numMines;

    BoardConfig();
};
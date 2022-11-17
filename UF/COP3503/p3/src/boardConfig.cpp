#include "boardConfig.h"

BoardConfig::BoardConfig() {
    std::ifstream cfgFile("boards/config.cfg");
    if (!cfgFile) throw std::runtime_error("Error: File not Found");
    cfgFile >> m_columns;
    cfgFile >> m_rows;
    cfgFile >> m_numMines;
}
from enum import Enum


class GameState(Enum):
    GAME_OVER = -1
    CONTINUE = 0
    ADVANCE = 1
    
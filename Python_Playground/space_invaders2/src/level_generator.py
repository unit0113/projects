import pygame
import random

from .enemy_factory import EnemyFactory
from .settings import (
    BASE_LEVEL_POINTS,
    LEVEL_POINTS_INCREASE,
    BASE_ENEMY_SPACING_MS,
    ENEMY_SPACING_LEVEL_DECREASE,
)


ENEMY_TYPES = ["bug"]


class LevelGenerator:
    def __init__(self) -> None:
        self.level = 1
        self.level_points = BASE_LEVEL_POINTS
        self.spacing = BASE_ENEMY_SPACING_MS

    def next_level(self) -> None:
        """Increase the level to be generated"""
        self.level += 1
        self.level_points *= LEVEL_POINTS_INCREASE
        self.spacing *= ENEMY_SPACING_LEVEL_DECREASE

    def set_level(self, level: int) -> None:
        """Set the level to be generated

        Args:
            level (int): game level
        """
        self.level = level
        self.level_points = BASE_LEVEL_POINTS * (LEVEL_POINTS_INCREASE**level)
        self.spacing = BASE_ENEMY_SPACING_MS * (ENEMY_SPACING_LEVEL_DECREASE**level)

    def generate_level(self) -> None:
        """Generates the enemies that will spawn during the level"""
        self.enemy_queue = []
        enemy_factory = EnemyFactory(self.level, random.choice(ENEMY_TYPES))
        points_remaining = self.level_points
        spawn_time = BASE_ENEMY_SPACING_MS // 2

        while points_remaining > 0:
            enemy = enemy_factory.get_enemy()
            points_remaining -= enemy.get_points()
            self.enemy_queue.append((spawn_time, enemy))
            spawn_time += random.randrange(self.spacing // 2, int(self.spacing * 2))

        self.level_start_time = pygame.time.get_ticks()

    def spawn_enemy(self):
        """Get next enemy if spawn time is passed"""
        if (
            self.enemy_queue
            and pygame.time.get_ticks() > self.enemy_queue[0][0] + self.level_start_time
        ):
            return self.enemy_queue.pop(0)[1]

    def level_over(self) -> bool:
        """Returns whether there are still enemies to spawn

        Returns:
            bool: _description_
        """
        return not self.enemy_queue

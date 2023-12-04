import pygame
import random
import copy

from src.enemy import Enemy
from src.enemy_data import ENEMY_DATA, ENEMY_SPAWN_DATA
from src.settings import ENEMY_SPACING, ENEMY_SPACING_VARIANCE


class EnemyFactory:
    def __init__(
        self,
        enemy_images: dict[str, pygame.surface.Surface],
        waypoints: list[list[int, int]],
    ) -> None:
        self.enemy_images = enemy_images
        self.waypoints = waypoints

    def get_enemy(self, waypoints: list[list[int, int]], type: str) -> Enemy:
        return Enemy(
            waypoints,
            self.enemy_images[type],
            ENEMY_DATA[type]["health"],
            ENEMY_DATA[type]["speed"],
        )

    def get_enemies(self, level: int) -> pygame.sprite.Group:
        enemy_group = pygame.sprite.Group()
        level_data = ENEMY_SPAWN_DATA[level - 1]
        new_waypoints = copy.deepcopy(self.waypoints)
        types = []
        for type, number in level_data.items():
            if number > 0:
                types.extend([type] * number)
        random.shuffle(types)

        for type in types:
            enemy_group.add(self.get_enemy(new_waypoints, type))
            new_waypoints[0][1] -= ENEMY_SPACING / len(types) + random.randint(
                -ENEMY_SPACING_VARIANCE, ENEMY_SPACING_VARIANCE
            )

        return enemy_group

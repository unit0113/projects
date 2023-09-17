import pygame
import random

from enemy import Enemy
from settings import HEIGHT, START_NUM_ENEMY_SCORE, LEVEL_SCALE, KNIGHT_HP, KNIGHT_DAMAGE, KNIGHT_SPEED, GOBLIN_HP, GOBLIN_DAMAGE, GOBLIN_SPEED, P_GOBLIN_HP, P_GOBLIN_DAMAGE, P_GOBLIN_SPEED, R_GOBLIN_HP, R_GOBLIN_DAMAGE, R_GOBLIN_SPEED

class EnemyFactory:
    def __init__(self, sprites: dict[str, pygame.surface.Surface]) -> None:
        self.sprites = sprites
        self.start_x = -25
        self.end_x = -500
        self.min_y = HEIGHT // 2
        self.max_y = 5 * HEIGHT // 6
        self.enemy_builders = [self._get_goblin, self._get_knight, self._get_p_goblin, self._get_r_goblin]
        self.num_enemies_score = START_NUM_ENEMY_SCORE

    def get_enemies(self) -> pygame.sprite.Group:
        enemy_group = pygame.sprite.Group()
        enemy_value = 0

        while enemy_value < self.num_enemies_score:
            builder = random.choice(self.enemy_builders)
            enemy = builder(random.randint(self.end_x, self.start_x), random.randint(self.min_y, self.max_y))
            enemy_group.add(enemy)
            enemy_value += enemy.hp
        
        self.num_enemies_score *= LEVEL_SCALE
        return enemy_group

    def _get_knight(self, x: int, y: int) -> Enemy:
        return Enemy(KNIGHT_HP, self.sprites['knight'], x, y, KNIGHT_SPEED, KNIGHT_DAMAGE)
    
    def _get_goblin(self, x: int, y: int) -> Enemy:
        return Enemy(GOBLIN_HP, self.sprites['goblin'], x, y, GOBLIN_SPEED, GOBLIN_DAMAGE)
    
    def _get_p_goblin(self, x: int, y: int) -> Enemy:
        return Enemy(P_GOBLIN_HP, self.sprites['purple_goblin'], x, y, P_GOBLIN_SPEED, P_GOBLIN_DAMAGE)
    
    def _get_r_goblin(self, x: int, y: int) -> Enemy:
        return Enemy(R_GOBLIN_HP, self.sprites['red_goblin'], x, y, R_GOBLIN_SPEED, R_GOBLIN_DAMAGE)
    
    def reset(self) -> None:
        self.num_enemies_score = START_NUM_ENEMY_SCORE
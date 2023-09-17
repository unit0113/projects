import pygame

from enemy_factory import EnemyFactory
from settings import BULLET_DAMAGE

class EnemyManager:
    def __init__(self, assets: dict[str, pygame.surface.Surface]) -> None:
        self.assets = assets
        self.factory = EnemyFactory(assets)
        self.next_level()

    def update(self, dt: float, window: pygame.surface.Surface) -> None:
        self.enemy_group.update(dt, window)

    def take_bullets(self, bullet_group: pygame.sprite.Group) -> int:
        money = 0
        for enemy in self.enemy_group:
            if enemy.action != 'death' and pygame.sprite.spritecollide(enemy, bullet_group, True):
                money += enemy.take_hit(BULLET_DAMAGE)

        return money

    def damage_castle(self) -> int:
        damage = 0
        for enemy in self.enemy_group:
            damage += enemy.do_damage()

        return damage
    
    def level_complete(self) -> bool:
        return len(self.enemy_group) == 0
    
    def next_level(self) -> None:
        self.enemy_group = self.factory.get_enemies()

    def reset(self) -> None:
        self.factory.reset()
        self.next_level()

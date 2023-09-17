import pygame

from settings import STARTING_HEALTH, SHOOT_COOLDOWN, BULLET_SPEED
from bullet import Bullet
from tower import Tower


class Castle:
    def __init__(self, assets: dict[str, pygame.surface.Surface], x: int, y: int) -> None:
        self.health = STARTING_HEALTH
        self.max_health = STARTING_HEALTH
        self.sprites = assets

        self.img = self.sprites['castle_100']
        self.rect = self.img.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.shoot_timer = SHOOT_COOLDOWN
        self.bullet_start = (self.rect.x, self.rect.midleft[1])
        self.bullet_group = pygame.sprite.Group()
        self.tower_group = pygame.sprite.Group()

    def update(self, dt: float) -> None:
        # Update bullets
        self.bullet_group.update(dt)

        # Shoot
        self.shoot(dt)

    def shoot(self, dt: float) -> None:
        self.shoot_timer += dt
        if self.shoot_timer > SHOOT_COOLDOWN and pygame.mouse.get_pressed()[0] and pygame.mouse.get_pos()[1] > 50:
            self.shoot_timer = 0
            for tower in self.tower_group:
                self.bullet_group.add(Bullet(*tower.bullet_start, *pygame.mouse.get_pos(), BULLET_SPEED, self.sprites['bullet']))
            self.bullet_group.add(Bullet(*self.bullet_start, *pygame.mouse.get_pos(), BULLET_SPEED, self.sprites['bullet']))
    
    def take_damage(self, damage: int) -> bool:
        self.health -= damage
        ratio = self.health / self.max_health

        if self.health <= 0:
            return True

        # Update castle img as required
        if ratio < 0.25:
            self.img = self.sprites['castle_25']
            for tower in self.tower_group:
                tower.set_25()
        elif ratio < 0.50:
            self.img = self.sprites['castle_50']
            for tower in self.tower_group:
                tower.set_50()
        else:
            self.img = self.sprites['castle_100']
            for tower in self.tower_group:
                tower.set_100()

        return False

    def get_health_status(self) -> tuple[int, int]:
        return (self.health, self.max_health)
    
    def can_repair(self) -> bool:
        return self.health < self.max_health
    
    def repair(self) -> None:
        self.health = min(self.health + 250, self.max_health)

    def upgrade_armor(self) -> None:
        self.max_health += 250
        self.health += 250

    def add_tower(self, x: int, y: int) -> None:
        self.tower_group.add(Tower(self.sprites['tower'], x, y))

    def reset(self) -> None:
        self.health = STARTING_HEALTH
        self.max_health = STARTING_HEALTH
        self.img = self.sprites['castle_100']
        self.shoot_timer = SHOOT_COOLDOWN
        self.bullet_group.empty()
        self.tower_group.empty()

    def draw(self, window: pygame.surface.Surface) -> None:
        window.blit(self.img, self.rect)
        self.tower_group.draw(window)
        self.bullet_group.draw(window)

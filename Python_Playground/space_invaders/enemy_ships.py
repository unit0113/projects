import pygame
import random
import os
from ship import Ship
from settings import (WIDTH, HEIGHT, FPS, AI_BASE_FIRE_RATE, AI_BASE_FIRE_CHANCE, AI_BASE_SPEED, AI_BASE_DMG, AI_BASE_HEALTH,
                      AI_SHIELD_STRENGTH_PER_LEVEL, AI_SHIELD_REGEN_BASE, AI_BASE_LEVEL_UPGRADE, POINT_HEALTH_MULTIPLIER,
                      POINT_SPEED_MULTIPLIER, POINT_LASER_MULTIPLIER, POINT_SHIELD_MULTIPLIER)


class EvilSpaceShip(Ship):
    def __init__(self, level):
        super().__init__()
        self.level = level
        self.image = pygame.transform.scale(pygame.image.load(os.path.join(r'Python_Playground\space_invaders\Assets', 'spaceship_red.png')), self.ship_size).convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = pygame.Rect(random.randint(20, WIDTH - 20 - self.ship_size[0]), -10 - self.ship_size[1], self.image.get_width(), self.image.get_height())
        self.laser_image = pygame.image.load(os.path.join(r'Python_Playground\space_invaders\Assets', 'green_laser.png')).convert_alpha()

        self.max_health = int(AI_BASE_HEALTH * random.uniform(0.8, 1.2) * self.level_multiplier)
        self.health = self.max_health
        self.velocity = pygame.Vector2(0, int(AI_BASE_SPEED * random.uniform(0.8, 1.2) * self.level_multiplier))
        #self.speed = int(AI_BASE_SPEED * random.uniform(0.8, 1.2) * self.level_multiplier)
        self.shield_level = random.choice([0] * (30 - self.level) + [1] * (self.level // 2) + [2] * (self.level // 4) + [3] * (self.level // 6) + [4] * (self.level // 8) + [5] * (self.level // 10))
        self.laser_level = random.choice([0] * (30 - self.level) + [1] * (self.level // 5) + [2] * (self.level // 10))
        self.point_value = self.health * POINT_HEALTH_MULTIPLIER + self.velocity.magnitude() * POINT_SPEED_MULTIPLIER + self.laser_level * POINT_LASER_MULTIPLIER + self.shield_level * POINT_SHIELD_MULTIPLIER
        self.laser_timer = AI_BASE_FIRE_RATE * FPS
        self.base_damage = AI_BASE_DMG * self.level_multiplier
        self.shield_strength = self.max_shield_strength
    
    @property
    def level_multiplier(self):
        return (1 + AI_BASE_LEVEL_UPGRADE) ** (self.level - 1)

    @property
    def can_fire(self):
        return self.rect.y > 0 and self.rect.y < HEIGHT - 2* self.ship_size[1] and self.laser_timer > AI_BASE_FIRE_RATE * FPS

    @property
    def will_fire(self):
        return random.uniform(0, 10) < AI_BASE_FIRE_CHANCE * self.level_multiplier / FPS
    
    @property
    def max_shield_strength(self):
        return AI_SHIELD_STRENGTH_PER_LEVEL * self.shield_level

    @property
    def shield_regen(self):
        return AI_SHIELD_REGEN_BASE * self.level_multiplier / FPS

    @property
    def shield_cooldown_modifier(self):
        return self.level_multiplier

    def update(self):
        if self.shield_level:
            self.update_shield()

        self.laser_timer += 1
        self.rect.x += self.velocity.x // FPS
        self.rect.y += self.velocity.y // FPS

    def fire(self):
        if self.can_fire and self.will_fire:
            self.laser_timer = 0
            return self.laser_types[self.laser_level]()

    def take_hit(self, damage):
        self.health -= damage

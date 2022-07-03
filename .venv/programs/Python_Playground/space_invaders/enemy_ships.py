import pygame
import random
import os
from ship import Ship
from space_invaders import WIDTH, HEIGHT, FPS


AI_BASE_FIRE_RATE = 25
AI_BASE_FIRE_CHANCE = 5
AI_BASE_SPEED = 120
AI_BASE_DMG = 20
AI_BASE_HEALTH = 100
AI_SHIELD_STRENGTH_PER_LEVEL = 25
AI_SHIELD_REGEN_BASE = 10
AI_BASE_LEVEL_UPGRADE = 1.1
POINT_HEALTH_MULTIPLIER = 10
POINT_SPEED_MULTIPLIER = 1
POINT_LASER_MULTIPLIER = 500
POINT_SHIELD_MULTIPLIER = 250


class EvilSpaceShip(Ship):
    def __init__(self, level):
        super().__init__()
        self.level = level
        self.image = pygame.transform.scale(pygame.image.load(os.path.join(r'Python_Playground\space_invaders\Assets', 'spaceship_red.png')), self.ship_size).convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = pygame.Rect(random.randint(20, WIDTH - 20 - self.ship_size[0]), -10 - self.ship_size[1], self.image.get_width(), self.image.get_height())
        self.max_health = int(AI_BASE_HEALTH * random.uniform(0.8, 1.2) * self.level_multiplier)
        self.health = self.max_health
        self.speed = int(AI_BASE_SPEED * random.uniform(0.8, 1.2) * self.level_multiplier)
        self.shield_level = random.choice([0] * (10 - self.level) + [1] * (self.level // 2) + [2] * (self.level // 4) + [3] * (self.level // 6) + [4] * (self.level // 8) + [5] * (self.level // 10))
        self.point_value = self.health * POINT_HEALTH_MULTIPLIER + self.speed * POINT_SPEED_MULTIPLIER + self.laser_type_current_index * POINT_LASER_MULTIPLIER + self.shield_level * POINT_SHIELD_MULTIPLIER
        self.laser_timer = AI_BASE_FIRE_RATE * FPS / 60
        self.base_damage = AI_BASE_DMG * self.level_multiplier
        self.shield_strength = self.max_shield_strength
    
    @property
    def level_multiplier(self):
        return 1 * AI_BASE_LEVEL_UPGRADE ** (self.level - 1)

    @property
    def can_fire(self):
        return self.rect.y > 0 and self.rect.y < HEIGHT - 2* self.ship_size[1] and self.laser_timer > AI_BASE_FIRE_RATE

    @property
    def will_fire(self):
        return random.uniform(0, 10) < AI_BASE_FIRE_CHANCE * self.level_multiplier / FPS
    
    @property
    def max_shield_strength(self):
        return AI_SHIELD_STRENGTH_PER_LEVEL * self.shield_level * self.level_multiplier

    @property
    def shield_regen(self):
        return AI_SHIELD_REGEN_BASE * self.level_multiplier / FPS

    def update(self):
        if self.shield_level:
            self.update_shield()

        self.laser_timer += 1
        self.rect.y += self.speed // FPS

    def fire(self):
        if self.can_fire and self.will_fire:
            self.laser_timer = 0
            return self.laser_types[self.laser_type_current_index]()

    def take_hit(self, damage):
        self.health -= damage

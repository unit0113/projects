import pygame
from ship import SHIP_SIZE
from space_invaders import FPS


SHIELD_SIZE = 1.25
SHIELD_STRENGTH_PER_LEVEL = 25
MAX_SHIELD_LEVEL = 5
SHIELD_REGEN_BASE = 10 / FPS
SHIELD_POST_HIT_COOLDOWN = 5 * FPS

SHIELD_BLUE = (63, 94, 249)


class Shields:
    def __init__(self, shield_level=0):
        self.shield_radius = SHIELD_SIZE * SHIP_SIZE[0]
        self.shield_levels_strength = [SHIELD_STRENGTH_PER_LEVEL * num for num in range(MAX_SHIELD_LEVEL + 1)]
        self.shield_level = shield_level
        self.shield_strength = self.max_shield_strength
        self.cooldown_timer = 0
        self.mask = pygame.mask.from_surface(self.create_shield_mask())

    @property
    def max_shield_strength(self):
        return self.shield_levels_strength[self.shield_level]

    @property
    def shields_up(self):
        return self.shield_strength > 0

    @property
    def can_regen(self):
        return self.cooldown_timer > SHIELD_POST_HIT_COOLDOWN

    @property
    def shield_regen(self):
        pass

    def create_shield_mask(self):
        shape_surf = pygame.Surface((self.shield_radius * 2, self.shield_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, SHIELD_BLUE, (self.shield_radius, self.shield_radius), self.shield_radius)
        return shape_surf

    def collision(self, x, y, laser):
        return x < laser.rect.x + self.shield_radius and x + laser.rect.width > laser.rect.x and y < laser.rect.y + self.shield_radius and laser.rect.height + y > laser.rect.y
        
    def take_hit(self, damage):
        self.shield_strength = max(0, self.shield_strength - damage)

    def update(self):
        self.cooldown_timer += 1
        if self.can_regen and self.shield_strength < self.max_shield_strength:
            self.shield_strength = max(self.shield_strength + self.shield_regen, self.max_shield_strength)

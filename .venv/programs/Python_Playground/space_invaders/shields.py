import pygame
from ship import SHIP_SIZE
from space_invaders import FPS

SHIELD_STRENGTH_PER_LEVEL = 25
MAX_SHIELD_LEVEL = 5
SHIELD_REGEN_BASE = 10 / FPS
SHIELD_POST_HIT_COOLDOWN = 5 * FPS

SHIELD_BLUE = (63, 94, 249)


class Shields:
    def __init__(self, shield_level=0):
        self.shield_levels_strength = [SHIELD_STRENGTH_PER_LEVEL * num for num in range(MAX_SHIELD_LEVEL + 1)]
        self.shield_level = shield_level
        self.shield_strength = self.max_shield_strength
        self.cooldown_timer = 0
        self.mask = pygame.mask.from_surface(self.create_mask())

    @property
    def max_shield_strength(self):
        return self.shield_levels_strength[self.shield_level]

    @property
    def shields_up(self):
        return self.shield_strength > 0

    @property
    def can_regen(self):
        return self.time > SHIELD_POST_HIT_COOLDOWN

    def create_mask(self):
        radius = int(1.5 * SHIP_SIZE[0])
        shape_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, SHIELD_BLUE, (radius, radius), radius)
        return shape_surf

    def collision(self, laser):
        pass
        #circle_x < rect_x + circle_width and circle_x + rect_width > rect_x and circle_y < rect_y + circle_height and rect_height + circle_y > rect_y
        
    def take_hit(self, damage):
        self.shield_strength = max(0, self.shield_strength - damage)

    def update(self):
        pass
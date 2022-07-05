import pygame
import random
from abc import ABC, abstractmethod, abstractproperty
from lasers import Laser, LASER_SIZE
from space_invaders import FPS


SHIP_SIZE = (50, 50)
SHIELD_SIZE = 0.8
SHIELD_STRENGTH_PER_LEVEL = 25
MAX_SHIELD_LEVEL = 5
SHIELD_POST_HIT_COOLDOWN = 5

SHIELD_BLUE_INNER = (63, 94, 249, 50)
SHIELD_BLUE_OUTER = (63, 94, 249, 200)


class Ship(ABC):
    def __init__(self):
        self.ship_size = SHIP_SIZE

        # Init lasers
        self.laser_types = [self.laser1, self.laser2, self.laser3]
        self.laser_type_dmg_multipliers = [1, 0.65, 0.5]
        self.laser_type_current_index = 0

        # Init shields
        self.shield_radius = SHIELD_SIZE * SHIP_SIZE[0]
        self.shield_level = 0
        self.shield_strength = 0
        self.shield_cooldown_timer = 0
        self.shield_mask = pygame.mask.from_surface(self.create_shield_mask())


    @property
    def damage(self):
        return self.base_damage * random.uniform(0.5, 1.5) * self.laser_type_dmg_multipliers[self.laser_type_current_index]

    @property
    def is_dead(self):
        return self.health <= 0

    @abstractproperty
    def max_shield_strength(self):
        pass
    
    @property
    def shields_up(self):
        return self.shield_strength > 0

    @property
    def can_regen(self):
        return self.shield_cooldown_timer > (SHIELD_POST_HIT_COOLDOWN / self.shield_cooldown_modifier) * FPS

    @abstractproperty
    def shield_cooldown_modifier(self):
        pass

    @abstractproperty
    def shield_regen(self):
        pass

    def create_shield_mask(self):
        self.shield_surf = pygame.Surface((2 * self.shield_radius, 2 * self.shield_radius), pygame.SRCALPHA)
        self.shield_rect = pygame.draw.circle(self.shield_surf, SHIELD_BLUE_INNER, (self.shield_radius, self.shield_radius), self.shield_radius)        # pygame.draw returns rect object, used for laser collision
        return self.shield_surf

    def get_center_points(self):
        # Shield surface is 0, 0 in top left corner, need to account for that when centering drawing of shields
        x = self.rect.x + self.ship_size[0] // 2 - self.shield_radius
        y = self.rect.y + self.ship_size[1] // 2 - self.shield_radius
        return x, y

    def shield_take_hit(self, damage):
        self.shield_strength = max(0, self.shield_strength - damage)
        self.shield_cooldown_timer = 0

    def update_shield(self):
        self.shield_cooldown_timer += 1
        self.shield_rect.center = self.rect.center      # Recenter shield rect for collision detection with colliderect
        if self.can_regen and self.shield_strength < self.max_shield_strength:
            self.shield_strength = max(self.shield_strength + self.shield_regen, self.max_shield_strength)

    def draw_shield(self, window):
        # Draw inner
        pygame.draw.circle(self.shield_surf, SHIELD_BLUE_INNER, self.get_center_points(), self.shield_radius)
        window.blit(self.shield_surf, self.get_center_points())
        # Draw outer
        pygame.draw.circle(self.shield_surf, SHIELD_BLUE_OUTER, (self.shield_radius, self.shield_radius), self.shield_radius, 2 * self.shield_level)
        window.blit(self.shield_surf, self.get_center_points())

    def laser1(self):
        # Single centerline laser
        laser = Laser(self.rect.x + self.image.get_width() // 2 - LASER_SIZE[0] // 2, self.rect.y, self.damage)
        return [laser]

    def laser2(self):
        # Dual wingtip lasers
        laser1 = Laser(self.rect.x, self.rect.y + 15, self.damage)
        laser2 = Laser(self.rect.x + SHIP_SIZE[0] - LASER_SIZE[0], self.rect.y + 15, self.damage)
        return [laser1, laser2]

    def laser3(self):
        # Centerline plus dual wingtip lasers
        laser1 = Laser(self.rect.x + self.image.get_width() // 2 - LASER_SIZE[0] // 2, self.rect.y, self.damage)
        laser2 = Laser(self.rect.x, self.rect.y + 15, self.damage)
        laser3 = Laser(self.rect.x + SHIP_SIZE[0] - LASER_SIZE[0], self.rect.y + 15, self.damage)
        return [laser1, laser2, laser3]

    def draw(self, window):
        window.blit(self.image, (self.rect.x, self.rect.y))
        if self.shield_strength:
            self.draw_shield(window)

    @abstractmethod
    def fire(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def take_hit(self):
        pass

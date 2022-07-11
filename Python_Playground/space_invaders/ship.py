import pygame
import random
from abc import ABC, abstractmethod, abstractproperty
from lasers import Laser, MiniGunLaser, SideLaser
from missiles import Missile
from settings import LASER_SIZE, MINIGUN_LASER_SIZE, MISSILE_SIZE, MISSILE_COOLDOWN, FPS, SHIP_SIZE, SHIELD_SIZE, SHIELD_POST_HIT_COOLDOWN, SHIELD_BLUE_INNER, SHIELD_BLUE_OUTER


class Ship(ABC):
    def __init__(self):
        self.ship_size = SHIP_SIZE

        # Init lasers
        self.laser_types = [self.laser1, self.laser2, self.laser3, self.laser4, self.laser5, self.laser6]
        self.laser_type_dmg_multipliers = [1, 0.65, 0.5, 0.4, 0.3, 0.3]
        self.laser_type_cost_multipliers = [1, 1, 1, 0.25, 0.25, 0.25]
        self.laser_type_fire_rate_multipliers = [1, 1, 1, 0.25, 0.25, 0.25]
        self.laser_level = 0
        self.laser_image = None
        self.side_laser_types = [None, self.side_laser1, self.side_laser2, self.side_laser3, self.side_laser4, self.side_laser5]
        self.side_laser_level = 0

        # Init shields
        self.shield_radius = SHIELD_SIZE * SHIP_SIZE[0]
        self.shield_level = 0
        self.shield_strength = 0
        self.shield_cooldown_timer = 0
        self.shield_mask = pygame.mask.from_surface(self.create_shield_mask())

        # Init missiles
        self.missile_cooldown = 0
        self.missile_image = None

    @property
    def damage(self):
        return self.base_damage * random.uniform(0.5, 1.5) * self.laser_type_dmg_multipliers[self.laser_level]

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

    @property
    def can_fire_missile(self):
        return self.missile_cooldown > MISSILE_COOLDOWN * FPS

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
        laser = Laser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image)
        return [laser]

    def laser2(self):
        # Dual wingtip lasers
        laser1 = Laser(self.rect.x, self.rect.y + 15, self.damage, self.laser_image)
        laser2 = Laser(self.rect.x + SHIP_SIZE[0] - LASER_SIZE[0], self.rect.y + 15, self.damage, self.laser_image)
        return [laser1, laser2]

    def laser3(self):
        # Centerline plus dual wingtip lasers
        laser1 = Laser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image)
        laser2 = Laser(self.rect.x, self.rect.y + 15, self.damage, self.laser_image)
        laser3 = Laser(self.rect.x + SHIP_SIZE[0] - LASER_SIZE[0], self.rect.y + 15, self.damage, self.laser_image)
        return [laser1, laser2, laser3]

    def laser4(self):
        # Single centerline minigun
        laser = MiniGunLaser(self.rect.center[0] - MINIGUN_LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image)
        return [laser]

    def laser5(self):
        # Dual wingtip lasers
        laser1 = MiniGunLaser(self.rect.x, self.rect.y + 15, self.damage, self.laser_image)
        laser2 = MiniGunLaser(self.rect.x + SHIP_SIZE[0] - MINIGUN_LASER_SIZE[0], self.rect.y + 15, self.damage, self.laser_image)
        return [laser1, laser2]

    def laser6(self):
        # Centerline plus dual wingtip lasers
        laser1 = MiniGunLaser(self.rect.center[0] - MINIGUN_LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image)
        laser2 = MiniGunLaser(self.rect.x, self.rect.y + 15, self.damage, self.laser_image)
        laser3 = MiniGunLaser(self.rect.x + SHIP_SIZE[0] - MINIGUN_LASER_SIZE[0], self.rect.y + 15, self.damage, self.laser_image)
        return [laser1, laser2, laser3]

    def side_laser1(self):
        laser1 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -45)
        laser2 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 45)
        return [laser1, laser2]

    def side_laser2(self):
        laser1 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -90)
        laser2 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -45)
        laser3 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 45)
        laser4 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 90)
        return [laser1, laser2, laser3, laser4]

    def side_laser3(self):
        laser1 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -90)
        laser2 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -60)
        laser3 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -30)
        laser4 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 30)
        laser5 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 60)
        laser6 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 90)
        return [laser1, laser2, laser3, laser4, laser5, laser6]

    def side_laser4(self):
        laser1 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -90)
        laser2 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -80)
        laser3 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -45)
        laser4 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -15)
        laser5 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 15)
        laser6 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 45)
        laser7 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 80)
        laser8 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 90)
        return [laser1, laser2, laser3, laser4, laser5, laser6, laser7, laser8]

    def side_laser5(self):
        laser1 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -90)
        laser2 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -80)
        laser3 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -45)
        laser4 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -15)
        laser5 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, -10)
        laser6 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 10)
        laser7 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 15)
        laser8 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 45)
        laser9 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 80)
        laser10 = SideLaser(self.rect.center[0] - LASER_SIZE[0] // 2, self.rect.center[1], self.damage, self.laser_image, 90)
        return [laser1, laser2, laser3, laser4, laser5, laser6, laser7, laser8, laser9, laser10]

    def fire_missile(self):
        if self.can_fire_missile:
            self.missile_cooldown = 0
            return [Missile(self.rect.x + self.image.get_width() // 2 - MISSILE_SIZE[0] // 2, self.rect.y + self.image.get_height() // 4, self.missile_damage, self.missile_image)]

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

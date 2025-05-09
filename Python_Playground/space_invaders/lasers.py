import pygame
import math
from settings import HEIGHT, LASER_SIZE, MINIGUN_LASER_SIZE


class Laser:
    def __init__(self, x, y, damage, laser_image):
        self.damage = damage
        self.image = laser_image
        self.image = pygame.transform.scale(self.image, LASER_SIZE)
        self.rect = pygame.Rect(x, y, self.image.get_width(), self.image.get_height())
        self.mask = pygame.mask.from_surface(self.image)

    @property
    def is_off_screen(self):
        return self.rect.y > HEIGHT or self.rect.y < -LASER_SIZE[1]

    def update(self, movement):
        self.rect.y += movement

    def draw(self, window):
        window.blit(self.image, (self.rect.x, self.rect.y))


class MiniGunLaser(Laser):
    def __init__(self, x, y, damage, laser_image):
        super().__init__(x, y, damage, laser_image)
        self.image = pygame.transform.scale(self.image, MINIGUN_LASER_SIZE)
        self.rect = pygame.Rect(x, y, self.image.get_width(), self.image.get_height())
        self.mask = pygame.mask.from_surface(self.image)


class SideLaser(Laser):
    def __init__(self, x, y, damage, laser_image, angle):
        super().__init__(x, y, damage, laser_image)
        self.rect.center = (x, y)
        self.image = pygame.transform.rotate(self.image, angle)
        self.angle = math.radians(angle + 90)  # Rotate so up is 0 deg and convert to radians

    def update(self, movement):
        self.rect.x -= math.cos(self.angle) * movement
        self.rect.y += math.sin(self.angle) * movement

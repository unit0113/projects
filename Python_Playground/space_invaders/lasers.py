import pygame
from settings import HEIGHT, LASER_SIZE, MINIGUN_LASER_SIZE


class Laser:
    def __init__(self, x, y, damage):
        self.damage = damage
        self.rect = pygame.Rect(x, y, *LASER_SIZE)
        self.mask = pygame.mask.Mask(LASER_SIZE, True)

    @property
    def is_off_screen(self):
        return self.rect.y > HEIGHT or self.rect.y < -LASER_SIZE[1]

    def update(self, movement):
        self.rect.y += movement

    def draw(self, window, color):
        pygame.draw.rect(window, color, self.rect)


class MiniGunLaser(Laser):
    def __init__(self, x, y, damage):
        self.damage = damage
        self.rect = pygame.Rect(x, y, *MINIGUN_LASER_SIZE)
        self.mask = pygame.mask.Mask(MINIGUN_LASER_SIZE, True)

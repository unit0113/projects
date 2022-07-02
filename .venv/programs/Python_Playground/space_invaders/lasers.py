import pygame
from space_invaders import HEIGHT


LASER_SIZE = (6, 30)


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
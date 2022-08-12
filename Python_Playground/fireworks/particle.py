import pygame
import random
from settings import GRAVITY, FPS


MASS_DECAY = 0.1


class Particle:
    def __init__(self, x: int, y: int, x_vel: int, y_vel: int, mass: int, color: tuple):
        self.x = x
        self.y = y
        self.vel = pygame.Vector2(x_vel, y_vel)
        self.mass = mass
        self.color = color

    @property
    def is_decayed(self):
        return self.mass < 0

    def update(self):
        self.x += self.vel.x
        self.x += self.vel.y
        self.vel.y += GRAVITY / FPS
        self.mass -= MASS_DECAY

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), int(self.mass))
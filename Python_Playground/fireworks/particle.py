import pygame
import random
from settings import GRAVITY, FPS


MASS_DECAY_SHORT = 0.15
MASS_DECAY = 0.1
MASS_DECAY_LONG = 0.05


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
        self.y += self.vel.y
        self.vel.y += (GRAVITY / FPS) * random.uniform(1, 1.1)
        self.mass -= MASS_DECAY_LONG

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), int(self.mass))
        
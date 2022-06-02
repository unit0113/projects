import random
import numpy as np
import pygame


# Color Palletes
SAND = (194, 178, 128)
ROCK = (73, 60, 60)
WOOD = (164, 116, 73)
GRASS = (0, 154, 23)

# Cave generation constants
FILL_PERCENT = 0.8

class Cave:
    def __init__(self, width, height, window):
        self.width = width
        self.height = height
        self.window = window
        self.ground_color = random.choice([SAND, GRASS])
        self.wall_color = random.choice([WOOD, ROCK])
        self.cave = np.random.rand(self.width, self.height, 3)
        self.cave = np.where(self.cave < FILL_PERCENT, self.ground_color, self.wall_color)


    def draw(self):
        surface = pygame.pixelcopy.make_surface(self.cave)
        self.window.blit(surface, (0,0))
        


# Tests

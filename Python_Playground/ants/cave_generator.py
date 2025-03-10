from enum import Enum
import random
import numpy as np
import pygame


class Terrain(Enum):
    GROUND = 0
    WALL = 1
    
# Color Palletes
SAND = (174, 158, 108)
ROCK = (73, 60, 60)
WOOD = (164, 116, 73)
GRASS = (3, 128, 24)

# Cave generation constants
SCALE_FACTOR = 4
FILL_PERCENT = 0.625
WALL_THRESHOLD = 8
OUTER_WALL_THICKNESS = 10
SMOOTHNESS = 7
MASK_SIZE = 2

class Cave:
    def __init__(self, width, height, window):
        self.width = width
        self.height = height
        self.window = window
        self.ground_color = random.choice([SAND, GRASS])
        self.wall_color = random.choice([WOOD, ROCK])

        # Initial cave generation
        self.cave = np.random.rand(self.width // SCALE_FACTOR, self.height // SCALE_FACTOR)
        self.cave = np.where(self.cave < FILL_PERCENT, Terrain.GROUND, Terrain.WALL)

        # Build outer walls
        self.set_cave_walls()

        # Smooth cave via cellular automata
        for _ in range(SMOOTHNESS):
            self.smooth_cave()

        # Convert cave from black/white to color
        self.finalize()

    def set_cave_walls(self):
        # Left/right walls
        self.cave[:OUTER_WALL_THICKNESS, :] = Terrain.WALL
        self.cave[-OUTER_WALL_THICKNESS:, :] = Terrain.WALL

        # Top/bottom walls
        self.cave[:, :OUTER_WALL_THICKNESS] = Terrain.WALL
        self.cave[:, -OUTER_WALL_THICKNESS:] = Terrain.WALL

    def smooth_cave(self):
        cave_copy = np.copy(self.cave)
        for row in range(OUTER_WALL_THICKNESS, self.cave.shape[0] - OUTER_WALL_THICKNESS):
            for col in range(OUTER_WALL_THICKNESS, self.cave.shape[1] - OUTER_WALL_THICKNESS):
                wall_count = self.get_surrounding_wall_count(row, col, cave_copy)
                if wall_count > WALL_THRESHOLD:
                    self.cave[row][col] = Terrain.WALL
                elif wall_count < WALL_THRESHOLD - MASK_SIZE // 2:
                    self.cave[row][col] = Terrain.GROUND

    def get_surrounding_wall_count(self, row, col, cave):
        check_array = cave[row-MASK_SIZE:row+MASK_SIZE+1, col-MASK_SIZE:col+MASK_SIZE+1]
        check_array[1][1] = Terrain.GROUND
        wall_count = np.count_nonzero(check_array == Terrain.WALL)

        return wall_count

    def finalize(self):
        self.resize()
        self.cave = np.repeat(self.cave[:, :, np.newaxis], 3, axis=2)
        self.cave = np.where(self.cave == Terrain.WALL, self.wall_color, self.ground_color)
        self.surface = pygame.pixelcopy.make_surface(self.cave)

    def resize(self):
        cave_copy = np.copy(self.cave)
        self.cave = np.resize(self.cave, (self.width, self.height))
        self.cave.fill(Terrain.WALL)
        for row in range(cave_copy.shape[0]):
            for col in range(cave_copy.shape[1]):
                self.cave[row*SCALE_FACTOR:(row+1)*SCALE_FACTOR, col*SCALE_FACTOR:(col+1)*SCALE_FACTOR] = cave_copy[row][col]
        self.smooth_cave()

    def draw(self):
        self.window.blit(self.surface, (0,0))


# Tests
cave = Cave(100, 100, None)
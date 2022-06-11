from enum import Enum
import random
import numpy as np
import pygame


class Terrain(Enum):
    GROUND = 0
    WALL = 1
    
# Color Palletes
SAND = (194, 178, 128)
ROCK = (73, 60, 60)
WOOD = (164, 116, 73)
GRASS = (0, 154, 23)

# Cave generation constants
FILL_PERCENT = 0.5
WALL_THRESHOLD = 4
OUTER_WALL_THICKNESS = 10
SMOOTHNESS = 5

class Cave:
    def __init__(self, width, height, window):
        self.width = width
        self.height = height
        self.window = window
        self.ground_color = random.choice([SAND, GRASS])
        self.wall_color = random.choice([WOOD, ROCK])

        # Initial cave generation
        self.cave = np.random.rand(self.width, self.height)
        self.cave = np.where(self.cave < FILL_PERCENT, Terrain.GROUND, Terrain.WALL)

        # Smooth cave via cellular automata
        self.set_cave_walls()
        for _ in range(SMOOTHNESS):
            self.smooth_cave()

        # Convert cave from black/white to color
        self.cave = np.repeat(self.cave[:, :, np.newaxis], 3, axis=2)
        self.cave = np.where(self.cave == Terrain.WALL, self.wall_color, self.ground_color)

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
                wall_count = self.get_surrounding_wall_count(row, col)
                if wall_count > WALL_THRESHOLD:
                    cave_copy[row][col] = Terrain.WALL
                elif wall_count < WALL_THRESHOLD:
                    cave_copy[row][col] = Terrain.WALL
        
        self.cave = np.copy(cave_copy)

    def get_surrounding_wall_count(self, row, col):
        # np.sum(self.cave[row-1:row+2, col-1:col+2]) - self.cave[row, col]
        check_array = self.cave[row-1:row+2, col-1:col+2]
        check_array[1][1] = 0
        wall_count = np.count_nonzero(check_array == Terrain.WALL)

        return wall_count

    def draw(self):
        surface = pygame.pixelcopy.make_surface(self.cave)
        self.window.blit(surface, (0,0))
        


# Tests
cave = Cave(3440, 1440, None)
#print(cave.cave)
#print(cave.surrounding_cell_mask)
#print(cave.get_surrounding_wall_count(50, 50))

import random
import numpy as np
import pygame


# Color Palletes
SAND = (194, 178, 128)
ROCK = (73, 60, 60)
WOOD = (164, 116, 73)
GRASS = (0, 154, 23)

# Cave generation constants
FILL_PERCENT = 0.5
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
        self.cave = np.random.rand(self.width, self.height, 3)
        self.cave = np.where(self.cave < FILL_PERCENT, self.ground_color, self.wall_color)

        # Create mask for finding walls
        self.surrounding_cell_mask = np.ones((3, 3, 3))
        self.surrounding_cell_mask[1][1] = 0
        self.surrounding_cell_mask = np.where(self.surrounding_cell_mask == 1, self.wall_color, 0)

        self.set_cave_walls()
        for _ in range(SMOOTHNESS):
            self.smooth_cave()


    def set_cave_walls(self):
        # Left/right walls
        self.cave[:OUTER_WALL_THICKNESS, :, :] = self.wall_color
        self.cave[-OUTER_WALL_THICKNESS:, :, :] = self.wall_color

        # Top/bottom walls
        self.cave[:, :OUTER_WALL_THICKNESS, :] = self.wall_color
        self.cave[:, -OUTER_WALL_THICKNESS:, :] = self.wall_color


    def smooth_cave(self):
        for row in range(OUTER_WALL_THICKNESS, self.cave.shape[0] - OUTER_WALL_THICKNESS):
            for col in range(OUTER_WALL_THICKNESS, self.cave.shape[1] - OUTER_WALL_THICKNESS):
                pass

    
    def get_surrounding_wall_count(self, row, col):
        check_array = self.cave[row-1:row+1, col-1:col+1, :]
        check_array[1][1] = 0
        wall_count = np.count_nonzero(check_array == self.wall_color)

        return wall_count


    def draw(self):
        surface = pygame.pixelcopy.make_surface(self.cave)
        self.window.blit(surface, (0,0))
        


# Tests
cave = Cave(100, 100, None)
#print(cave.cave)
#print(cave.surrounding_cell_mask)
print(cave.get_surrounding_wall_count(50, 50))

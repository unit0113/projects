from tkinter import LEFT
from prometheus_client import Enum
import pygame
import os
import random
from enum import Enum
from PIL import Image


TILE_EDGE_SIZE = 100
TILE_SIZE = (TILE_EDGE_SIZE, TILE_EDGE_SIZE)
NUM_TILES = 10
WIDTH = NUM_TILES * TILE_SIZE[0]
HEIGHT = NUM_TILES * TILE_SIZE[1]
POST_COLLAPSE_WAIT_TIME_MS = 250

class Directions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


OPPOSITES = {Directions.UP: Directions.DOWN,
             Directions.RIGHT: Directions.LEFT,
             Directions.DOWN: Directions.UP,
             Directions.LEFT: Directions.RIGHT}


"""TODO
Store list of actions for reversal
function for Iding valid connections
"""


def is_valid_img(main_img, direction, other_img):
    main_img = Image.open(main_img)
    main_img_pic = main_img.load()
    other_img = Image.open(other_img)
    other_img_pic = other_img.load()

    if direction == Directions.LEFT or direction == Directions.RIGHT:
        edge_len = main_img.size[1]
    else:
        edge_len = main_img.size[0]

    matches = 0

    if direction == Directions.RIGHT:
        for y in range(edge_len):
            if main_img_pic[-1, y] == other_img_pic[0, y]:
                matches += 1

    elif direction == Directions.LEFT:
        for y in range(edge_len):
            if main_img_pic[0, y] == other_img_pic[-1, y]:
                matches += 1

    elif direction == Directions.UP:
        for x in range(edge_len):
            if main_img_pic[x, 0] == other_img_pic[x, -1]:
                matches += 1

    elif direction == Directions.DOWN:
        for x in range(edge_len):
            if main_img_pic[x, -1] == other_img_pic[x, 0]:
                matches += 1

    return (matches / edge_len) > 0.95


def get_all_tile_images(tile_options_path):
    possibilites = os.listdir(tile_options_path)
    final_possibilites = [Image.open(os.path.join(tile_options_path, possibilites[0]))]

    for possibility in possibilites[1:]:
        poss_img = Image.open(os.path.join(tile_options_path, possibility))
        for rotation in range(1,4):
            final_possibilites.append(poss_img.rotate(90 * rotation))

    return final_possibilites


def pilImageToSurface(pilImage):
    return pygame.image.fromstring(pilImage.tobytes(), pilImage.size, pilImage.mode).convert()








class Cell:
    def __init__(self, x, y, tile_options_path):
        self.tile_options_path = tile_options_path
        self.rect = pygame.Rect(x, y, *TILE_SIZE)
        self.collapsed = False
        self.possibilities = os.listdir(self.tile_options_path)
        self.base_image_path = os.path.join(self.tile_options_path, self.possibilities[0])
        self.image = pygame.image.load(self.base_image_path)
        self.possibilities.pop(0)

    @property
    def entrophy(self):
        return len(self.possibilities)

    def collapse(self):
        img = random.choice(self.possibilities)
        self.image = pygame.image.load(os.path.join(self.tile_options_path, img))
        self.possibilities.remove(img)
        self.collapsed = True

    def uncollapse(self):
        self.collapsed = False
        self.image = pygame.image.load(self.base_image_path)


    def draw(self, window):
        pass

    def __lt__(self, other):
        return self.entrophy < other.entrophy

    def __eq__(self, other):
        return self.entrophy == other.entrophy


class TileManager:
    def __init__(self):
        self.tile_options_path = self.select_tile_options()
        self.window = initialize_pygame()
        self.create_grid()

    def create_grid(self):
        pass

    def draw(self):
        pygame.display.update()

    def collapse(self):
        pygame.time.wait(POST_COLLAPSE_WAIT_TIME_MS)

    def select_tile_options(self):
        tiles_path = r'Python_Playground\wave_function_collapse\tiles'
        tile_options = os.listdir(tiles_path)
        print(*tile_options, sep=', ')

        selection = None
        while selection not in tile_options:
            selection = input('Select a set of tiles: ')

        return os.path.join(tiles_path, selection)


                        

def initialize_pygame():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Space Invaders")
    pygame.font.init()

    return window


def main():
    tile_manager = TileManager()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_q]:
            pygame.quit()
            quit()

        if keys[pygame.K_r]:
            main()


        tile_manager.collapse()


if __name__ == "__main__":
    main()

import pygame
import os
import random


TILE_SIZE = (100, 100)
NUM_TILES = 10
WIDTH = NUM_TILES * TILE_SIZE[0]
HEIGHT = NUM_TILES * TILE_SIZE[1]
POST_COLLAPSE_WAIT_TIME_MS = 500


"""TODO
Store list of actions for reversal
function for Iding valid connections
"""


class Cell:
    def __init__(self, x, y, tile_options_path):
        self.rect = pygame.Rect(x, y, *TILE_SIZE)
        self.collapsed = False
        self.possibilities = os.listdir(tile_options_path)
        self.image = pygame.image.load(os.path.join(tile_options_path, self.possibilities[0]))

    @property
    def entrophy(self):
        return len(self.possibilities)

    def draw(self):
        pass

    def __lt__(self, other):
        return self.entrophy < other.entrophy

    def __eq__(self, other):
        return self.entrophy == other.entrophy


class TileManager:
    def __init__(self):
        self.tile_options_path = self.select_tile_options()
        self.window = initialize_pygame()

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

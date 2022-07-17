import pygame
import os
import random
from enum import Enum
from PIL import Image
import math

SCREEN_HEIGHT = 1440
TILE_EDGE_SIZE = 100
TILE_SIZE = (TILE_EDGE_SIZE, TILE_EDGE_SIZE)
NUM_TILES = (SCREEN_HEIGHT - 100) // TILE_EDGE_SIZE
WIDTH = NUM_TILES * TILE_SIZE[0]
HEIGHT = NUM_TILES * TILE_SIZE[1]
POST_COLLAPSE_WAIT_TIME_MS = 100
BLACK = (0, 0, 0)


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


def is_valid_match(main_img, direction, other_img):
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


class Tile:
    def __init__(self, x, y, tile_options_path):
        self.x = x
        self.y = y
        self.tile_options_path = tile_options_path
        self.rect = pygame.Rect(x, y, *TILE_SIZE)
        self.collapsed = False
        self.possibilities = self._get_all_tile_images(self.tile_options_path)
        self.base_image = self.possibilities[0]
        self.image = self.base_image
        self.image = pygame.transform.scale(self.image, TILE_SIZE)
        self.possibilities.pop(0)

    @property
    def entrophy(self):
        return len(self.possibilities)

    def _get_all_tile_images(self, tile_options_path):
        possibilites = os.listdir(tile_options_path)
        base_img = self._convert_pil_img_to_surface(Image.open(os.path.join(tile_options_path, possibilites[0])))
        final_possibilites = [base_img]

        for possibility in possibilites[1:]:
            poss_img = Image.open(os.path.join(tile_options_path, possibility))
            for rotation in range(1,4):
                image = self._convert_pil_img_to_surface(poss_img.rotate(90 * rotation))
                final_possibilites.append(image)

        return final_possibilites

    
    def _convert_pil_img_to_surface(self, pilImage):
        return pygame.image.fromstring(pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

    def collapse(self):
        new_image = random.choice(self.possibilities)
        self.image = pygame.transform.scale(new_image, TILE_SIZE)
        self.possibilities.remove(new_image)
        self.collapsed = True

    def uncollapse(self):
        self.collapsed = False
        self.image = pygame.image.load(self.base_image_path)
        self.image = pygame.transform.scale(self.image, TILE_SIZE)

    def draw(self, window):
        window.blit(self.image, (self.x, self.y))

    def __lt__(self, other):
        return self.entrophy < other.entrophy

    def __eq__(self, other):
        return self.entrophy == other.entrophy

    def __str__(self):
        return f'Tile at ({self.x}, {self.y})'

    def __repr__(self):
        return self.__str__()


class TileManager:
    def __init__(self):
        self.tile_options_path = self._select_tile_options()
        self.window = self._initialize_pygame()
        self._create_grid()
    
    def _select_tile_options(self):
        tiles_path = r'Python_Playground\wave_function_collapse\tiles'
        tile_options = os.listdir(tiles_path)
        print(*tile_options, sep=', ')

        selection = None
        while selection not in tile_options:
            selection = input('Select a set of tiles: ')

        return os.path.join(tiles_path, selection)

    def _initialize_pygame(self):
        pygame.init()
        window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Wave Function Collapse")
        pygame.font.init()

        return window

    def _create_grid(self):
        self.grid = []
        x = y = 0

        for _ in range(NUM_TILES):
            row = []
            for _ in range(NUM_TILES):
                row.append(Tile(x, y, self.tile_options_path))
                x += TILE_SIZE[0]
            self.grid.append(row)
            y += TILE_SIZE[1]
            x = 0

    def draw(self):
        self.window.fill(BLACK)
        for row in self.grid:
            for tile in row:
                tile.draw(self.window)

        pygame.display.update()

    def collapse(self):
        tile_to_collapse = self._find_lowest_entrpohy()
        if tile_to_collapse[0]:
            tile_to_collapse[0].collapse()
            self._reduce_surrounding_entrophy(*tile_to_collapse)
            pygame.time.wait(POST_COLLAPSE_WAIT_TIME_MS)
        else:
            return True
    
    def _find_lowest_entrpohy(self):
        min_entrophy = math.inf
        low_tiles = []
        for row_index, row in enumerate(self.grid):
            for col_index, tile in enumerate(row):
                if not tile.collapsed:
                    if tile.entrophy < min_entrophy:
                        min_entrophy = tile.entrophy
                        low_tiles = [(tile, row_index, col_index)]
                    elif tile.entrophy == min_entrophy:
                        low_tiles.append((tile, row_index, col_index))

        if low_tiles:
            return random.choice(low_tiles)

        else:
            return None

    def _reduce_surrounding_entrophy(self, tile, row, col):
        pass



                        


def main():
    tile_manager = TileManager()
    complete = False

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

        tile_manager.draw()

        if not complete:
            complete = tile_manager.collapse()


if __name__ == "__main__":
    main()

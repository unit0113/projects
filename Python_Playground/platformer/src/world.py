import pygame

from .settings import TILE_SIZE, HEIGHT, PLAYER_MOVE_SPEED, PLAYER_JUMP_SPEED
from .tile import Tile
from .player import Player


TILE_TRANSLATOR = {1: 'dirt',
                   2: 'grass'}


class World:
    def __init__(self, images: dict[str: pygame.surface.Surface], player_sprites: dict[int: pygame.surface.Surface], window: pygame.surface.Surface, world_data: list[list[int]]) -> None:
        self.images = images
        self.window = window
        self.tiles = self.load_tiles(world_data)
        self.player = Player(100, HEIGHT - (TILE_SIZE + 80), player_sprites, window)

    def load_tiles(self, world_data: list[list[int]]) -> list[Tile]:
        tiles = []
        for row_index, row in enumerate(world_data):
            for col_index, tile_code in enumerate(row):
                if tile_code in TILE_TRANSLATOR.keys():
                    tiles.append(Tile(col_index * TILE_SIZE, row_index * TILE_SIZE, TILE_SIZE, self.images[TILE_TRANSLATOR[tile_code]], self.window))

        return tiles
    
    def update(self, dt: float, inputs: pygame.key.ScancodeWrapper) -> None:
        self.player.update(dt, inputs)

    def _check_player_colision(self) -> bool:
        pass

    def draw(self) -> None:
        # Draw background
        self.window.blit(self.images['sky'], (0,0))
        self.window.blit(self.images['sun'], (100,100))

        # Draw tiles
        for tile in self.tiles:
            tile.draw()

        # Draw player
        self.player.draw()

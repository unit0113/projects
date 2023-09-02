import pygame
import pickle
from os import path, listdir

from .settings import TILE_SIZE, HEIGHT
from .tile import Tile
from .player import Player
from .slime import Slime
from .lava import Lava
from .gate import Gate
from .game_state import GameState


TILE_TRANSLATOR = {1: 'dirt',
                   2: 'grass',
                   3: 'slime',
                   6: 'lava',
                   8: 'gate'
                   }

SLIME_OFFSET = 15
LAVA_OFFSET = TILE_SIZE // 2
GATE_OFFSET = TILE_SIZE // 2


class World:
    def __init__(self, images: dict[str: pygame.surface.Surface], player_sprites: dict[int: pygame.surface.Surface], window: pygame.surface.Surface, level: int) -> None:
        self.images = images
        self.window = window
        self.player_sprites = player_sprites
        self.tiles = []
        self.slime_group = pygame.sprite.Group()
        self.lava_group = pygame.sprite.Group()
        self.gate_group = pygame.sprite.Group()
        self.reset(level)

    def reset(self, level: int) -> None:
        world_data = self.load_world_data(level)
        self.tiles = []
        self.slime_group.empty()
        self.lava_group.empty()
        self.gate_group.empty()

        self.load_tiles(world_data)   
        self.player = Player(100, HEIGHT - (TILE_SIZE + 80), self.player_sprites, self.images['death'], self.window)
        self.game_over = False 

    def load_world_data(self, level: int) -> list[list[int]]:
        file_path = f'levels/level{level}_data'
        if path.exists(file_path):
            pickle_in = open(file_path, 'rb')
            return pickle.load(pickle_in)
        else:
            raise FileNotFoundError

    def load_tiles(self, world_data: list[list[int]]) -> None:
        for row_index, row in enumerate(world_data):
            for col_index, tile_code in enumerate(row):
                if tile_code in [1,2]:
                    self.tiles.append(Tile(col_index * TILE_SIZE, row_index * TILE_SIZE, TILE_SIZE, self.images[TILE_TRANSLATOR[tile_code]], self.window))
                elif tile_code == 3:
                    self.slime_group.add(Slime(col_index * TILE_SIZE, row_index * TILE_SIZE + SLIME_OFFSET, self.images[TILE_TRANSLATOR[tile_code]]))
                elif tile_code == 6:
                    self.lava_group.add(Lava(col_index * TILE_SIZE, row_index * TILE_SIZE + LAVA_OFFSET, self.images[TILE_TRANSLATOR[tile_code]]))
                elif tile_code == 8:
                    self.gate_group.add(Gate(col_index * TILE_SIZE, row_index * TILE_SIZE - GATE_OFFSET, self.images[TILE_TRANSLATOR[tile_code]]))
    
    def update(self, dt: float, inputs: pygame.key.ScancodeWrapper) -> GameState:
        self.player.update(dt, inputs, self.tiles)
        # Check for level advance
        if pygame.sprite.spritecollide(self.player, self.gate_group, False):
            return GameState.ADVANCE

        # Check death
        self.game_over = self.player.check_death(self.slime_group, self.lava_group, self.game_over)
        if not self.game_over:
            self.slime_group.update()
            return GameState.CONTINUE
        elif not self.player.death_animation_complete:
            return GameState.CONTINUE
        else:
            return GameState.GAME_OVER
        
    def advance_level(self) -> bool:
        return pygame.sprite.spritecollide(self.player, self.gate_group, False)

    def draw(self) -> None:
        # Draw background
        self.draw_background()

        # Draw tiles
        for tile in self.tiles:
            tile.draw()

        # Draw lava
        self.lava_group.draw(self.window)

        # Draw gate
        self.gate_group.draw(self.window)

        # Draw slimes
        self.slime_group.draw(self.window)

        # Draw player
        self.player.draw()

    def draw_background(self) -> None:
        # Draw background
        self.window.blit(self.images['sky'], (0,0))
        self.window.blit(self.images['sun'], (100,100))
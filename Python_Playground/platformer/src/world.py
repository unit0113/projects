import pygame

from .settings import TILE_SIZE, HEIGHT
from .tile import Tile
from .player import Player
from .slime import Slime
from .lava import Lava


TILE_TRANSLATOR = {1: 'dirt',
                   2: 'grass',
                   3: 'slime',
                   6: 'lava'
                   }

SLIME_OFFSET = 15
LAVA_OFFSET = TILE_SIZE // 2


class World:
    def __init__(self, images: dict[str: pygame.surface.Surface], player_sprites: dict[int: pygame.surface.Surface], window: pygame.surface.Surface, world_data: list[list[int]]) -> None:
        self.images = images
        self.window = window
        self.tiles, self.slime_group, self.lava_group = self.load_tiles(world_data)
        self.player = Player(100, HEIGHT - (TILE_SIZE + 80), player_sprites, images['death'], window)
        self.game_over = False

    def load_tiles(self, world_data: list[list[int]]) -> tuple[list[Tile], list[Slime]]:
        tiles = []
        slime_group = pygame.sprite.Group()
        lava_group = pygame.sprite.Group()
        for row_index, row in enumerate(world_data):
            for col_index, tile_code in enumerate(row):
                if tile_code in [1,2]:
                    tiles.append(Tile(col_index * TILE_SIZE, row_index * TILE_SIZE, TILE_SIZE, self.images[TILE_TRANSLATOR[tile_code]], self.window))
                elif tile_code == 3:
                    slime_group.add(Slime(col_index * TILE_SIZE, row_index * TILE_SIZE + SLIME_OFFSET, self.images[TILE_TRANSLATOR[tile_code]]))
                elif tile_code == 6:
                    lava_group.add(Lava(col_index * TILE_SIZE, row_index * TILE_SIZE + LAVA_OFFSET, self.images[TILE_TRANSLATOR[tile_code]]))

        return tiles, slime_group, lava_group
    
    def update(self, dt: float, inputs: pygame.key.ScancodeWrapper) -> bool:
        self.player.update(dt, inputs, self.tiles)
        self.game_over = self.player.check_death(self.slime_group, self.lava_group, self.game_over)
        if not self.game_over:
            self.slime_group.update()
            return False
        else:
            return self.player.death_animation_complete

    def _check_player_colision(self) -> bool:
        pass

    def draw(self) -> None:
        # Draw background
        self.window.blit(self.images['sky'], (0,0))
        self.window.blit(self.images['sun'], (100,100))

        # Draw tiles
        for tile in self.tiles:
            tile.draw()

        # Draw lava
        self.lava_group.draw(self.window)

        # Draw slimes
        self.slime_group.draw(self.window)

        # Draw player
        self.player.draw()

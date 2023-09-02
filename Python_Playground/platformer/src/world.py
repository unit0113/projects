import pygame
from pygame import mixer
import pickle
from os import path

from .settings import TILE_SIZE, HEIGHT, WHITE
from .tile import Tile
from .player import Player
from .slime import Slime
from .platform import Platform
from .lava import Lava
from .gate import Gate
from .coin import Coin
from .game_state import GameState


TILE_TRANSLATOR = {1: 'dirt',
                   2: 'grass',
                   3: 'slime',
                   4: 'platform',
                   5: 'platform',
                   6: 'lava',
                   7: 'coin',
                   8: 'gate'
                   }

SLIME_OFFSET = 15
LAVA_OFFSET = TILE_SIZE // 2
GATE_OFFSET = TILE_SIZE // 2


class World:
    def __init__(self, images: dict[str: pygame.surface.Surface], sounds: dict[str: mixer.Sound], player_sprites: dict[int: pygame.surface.Surface], window: pygame.surface.Surface, level: int) -> None:
        self.images = images
        self.sounds = sounds
        self.window = window
        self.player_sprites = player_sprites
        self.score = 0
        self.score_font = pygame.font.SysFont('Baushaus 93', 40)
        self.game_over_font = pygame.font.SysFont('Baushaus 93', 70)
        self.tiles = []
        self.slime_group = pygame.sprite.Group()
        self.platform_group = pygame.sprite.Group()
        self.lava_group = pygame.sprite.Group()
        self.gate_group = pygame.sprite.Group()
        self.coin_group = pygame.sprite.Group()
        self.reset(level)

    def reset(self, level: int) -> None:
        world_data = self.load_world_data(level)
        self.tiles = []
        self.slime_group.empty()
        self.platform_group.empty()
        self.lava_group.empty()
        self.gate_group.empty()
        self.coin_group.empty()
        self.coin_group.add(Coin(TILE_SIZE // 2, TILE_SIZE // 2, self.images['coin']))

        self.load_tiles(world_data)   
        self.player = Player(100, HEIGHT - (TILE_SIZE + 80), self.player_sprites, self.images['death'], self.sounds, self.window)
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
                elif tile_code == 4:
                    self.platform_group.add(Platform(col_index * TILE_SIZE, row_index * TILE_SIZE, self.images[TILE_TRANSLATOR[tile_code]], True))
                elif tile_code == 5:
                    self.platform_group.add(Platform(col_index * TILE_SIZE, row_index * TILE_SIZE, self.images[TILE_TRANSLATOR[tile_code]], False))
                elif tile_code == 6:
                    self.lava_group.add(Lava(col_index * TILE_SIZE, row_index * TILE_SIZE + LAVA_OFFSET, self.images[TILE_TRANSLATOR[tile_code]]))
                elif tile_code == 7:
                    self.coin_group.add(Coin(col_index * TILE_SIZE + TILE_SIZE // 2, row_index * TILE_SIZE + TILE_SIZE // 2, self.images[TILE_TRANSLATOR[tile_code]]))
                elif tile_code == 8:
                    self.gate_group.add(Gate(col_index * TILE_SIZE, row_index * TILE_SIZE - GATE_OFFSET, self.images[TILE_TRANSLATOR[tile_code]]))
    
    def update(self, dt: float, inputs: pygame.key.ScancodeWrapper) -> GameState:
        self.player.update(dt, inputs, self.tiles, self.platform_group)
        # Check for level advance
        if pygame.sprite.spritecollide(self.player, self.gate_group, False):
            return GameState.ADVANCE
        
        # Check coin collision
        if pygame.sprite.spritecollide(self.player, self.coin_group, True):
            self.sounds['coin'].play()
            self.score += 1

        # Check death
        self.game_over = self.player.check_death(self.slime_group, self.lava_group, self.game_over)
        if not self.game_over:
            self.slime_group.update()
            self.platform_group.update()
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

        # Draw platforms
        self.platform_group.draw(self.window)

        # Draw lava
        self.lava_group.draw(self.window)

        # Draw gate
        self.gate_group.draw(self.window)

        # Draw slimes
        self.slime_group.draw(self.window)

        # Draw coins
        self.coin_group.draw(self.window)

        # Draw score
        self.draw_text(f'x {self.score}', self.score_font, WHITE, TILE_SIZE, TILE_SIZE // 2)

        # Draw player
        self.player.draw()

    def draw_background(self) -> None:
        # Draw background
        self.window.blit(self.images['sky'], (0,0))
        self.window.blit(self.images['sun'], (100,100))

    def draw_text(self, text: str, font: pygame.font.Font, color: tuple[int, int, int], x: int, y: int) -> None:
        img = font.render(text, True, color)
        self.window.blit(img, (x, y - img.get_height() // 2))

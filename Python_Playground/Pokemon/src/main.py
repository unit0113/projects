from pytmx.util_pygame import load_pygame
from os.path import join

from settings import *
from sprites import *
from entities import *
from groups import *
from support import *


class Game:
    def __init__(self):
        pygame.init()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Pokemon")
        self.clock = pygame.time.Clock()

        # Groups
        self.all_sprites = AllSprites()

        self.import_assets()
        self.setup(self.tmx_maps["world"], "house")

    def import_assets(self):
        self.tmx_maps = {
            "world": load_pygame(join("data", "maps", "world.tmx")),
            "hospital": load_pygame(join("data", "maps", "hospital.tmx")),
        }

        self.overworld_frames = {
            "water": import_folder("graphics", "tilesets", "water"),
            "coast": coast_importer(24, 12, "graphics", "tilesets", "coast"),
            "characters": all_character_import("graphics", "characters"),
        }

    def setup(self, tmx_map, player_start_pos):
        # Terrain Tiles
        for layer in ["Terrain", "Terrain Top"]:
            for x, y, surf in tmx_map.get_layer_by_name(layer).tiles():
                Sprite((x * TILE_SIZE, y * TILE_SIZE), surf, self.all_sprites)

        # Objects
        for obj in tmx_map.get_layer_by_name("Objects"):
            Sprite((obj.x, obj.y), obj.image, self.all_sprites)

        # Entites
        for obj in tmx_map.get_layer_by_name("Entities"):
            if obj.name == "Player":
                if obj.properties["pos"] == player_start_pos:
                    self.player = Player(
                        pos=(obj.x, obj.y),
                        frames=self.overworld_frames["characters"]["player"],
                        groups=self.all_sprites,
                        facing_direction=obj.properties["direction"],
                    )
            else:
                Character(
                    pos=(obj.x, obj.y),
                    frames=self.overworld_frames["characters"][
                        obj.properties["graphic"]
                    ],
                    groups=self.all_sprites,
                    facing_direction=obj.properties["direction"],
                )

        # Water
        for obj in tmx_map.get_layer_by_name("Water"):
            for x in range(int(obj.x), int(obj.x + obj.width), TILE_SIZE):
                for y in range(int(obj.y), int(obj.y + obj.height), TILE_SIZE):
                    AnimatedSprite(
                        (x, y), self.overworld_frames["water"], self.all_sprites
                    )

        # Coast
        for obj in tmx_map.get_layer_by_name("Coast"):
            terrain = obj.properties["terrain"]
            side = obj.properties["side"]
            AnimatedSprite(
                (obj.x, obj.y),
                self.overworld_frames["coast"][terrain][side],
                self.all_sprites,
            )

    def run(self):
        while True:
            dt = self.clock.tick() / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            self.all_sprites.update(dt)
            self.display_surface.fill("black")
            self.all_sprites.draw(self.player.rect.center)
            pygame.display.update()


if __name__ == "__main__":
    game = Game()
    game.run()

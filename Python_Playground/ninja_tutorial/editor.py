import pygame

from src.tilemap import Tilemap
from src.utils import load_image, load_images
from src.settings import WIDTH, HEIGHT, FPS

RENDER_SCALE = 2.0


class Editor:
    def __init__(self) -> None:
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.display = pygame.Surface((WIDTH // 2, HEIGHT // 2))
        self.clock = pygame.time.Clock()

        pygame.display.set_caption("Ninja Game Level Editor")

        self._load_assets()
        self.movement = [0, 0, 0, 0]
        self.tile_map = Tilemap(self, tile_size=16)

        # Camera vars
        self.scroll = [0, 0]

        # Select tile
        self.tile_group = 0
        self.tile_variant = 0

        # Keys
        self.left_clicking = False
        self.right_clicking = False
        self.shift = False
        self.on_grid = True

        # Load if exists
        try:
            self.tile_map.load("map.json")
        except FileNotFoundError:
            pass

    def _load_assets(self) -> None:
        self.assets = {}

        self.tile_list = ["decor", "grass", "large_decor", "stone", "spawners"]
        for folder in self.tile_list:
            self.assets[folder] = load_images(f"tiles/{folder}")

        self.assets["background"] = load_image("background.png")

    def run(self) -> None:
        while True:
            self.clock.tick(FPS)
            mpos = pygame.mouse.get_pos()
            mpos = (mpos[0] / RENDER_SCALE, mpos[1] / RENDER_SCALE)
            tile_pos = (
                int((mpos[0] + self.scroll[0]) // self.tile_map.tile_size),
                int((mpos[1] + self.scroll[1]) // self.tile_map.tile_size),
            )

            # Event handler
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # Quit if escape is pressed
                        pygame.quit()
                        quit()
                    if event.key == pygame.K_r:
                        # game.reset()
                        pass
                    if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        self.movement[0] = 1
                    if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        self.movement[1] = 1
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        self.movement[2] = 1
                    if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        self.movement[3] = 1
                    if event.key == pygame.K_g:
                        self.on_grid = not self.on_grid
                    if event.key == pygame.K_t:
                        self.tile_map.autotile()
                    if event.key == pygame.K_o:
                        self.tile_map.save("map.json")
                    if event.key == pygame.K_LSHIFT:
                        self.shift = True
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        self.movement[0] = 0
                    if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        self.movement[1] = 0
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        self.movement[2] = 0
                    if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        self.movement[3] = 0
                    if event.key == pygame.K_LSHIFT:
                        self.shift = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.left_clicking = True
                        if not self.on_grid:
                            self.tile_map.offgrid_tiles.append(
                                {
                                    "type": self.tile_list[self.tile_group],
                                    "variant": self.tile_variant,
                                    "pos": (
                                        mpos[0] + self.scroll[0],
                                        mpos[1] + self.scroll[1],
                                    ),
                                }
                            )
                    if event.button == 3:
                        self.right_clicking = True
                    if self.shift:
                        if event.button == 4:
                            self.tile_variant = (self.tile_variant - 1) % len(
                                self.assets[self.tile_list[self.tile_group]]
                            )
                        if event.button == 5:
                            self.tile_variant = (self.tile_variant + 1) % len(
                                self.assets[self.tile_list[self.tile_group]]
                            )
                    else:
                        if event.button == 4:
                            self.tile_group = (self.tile_group - 1) % len(
                                self.tile_list
                            )
                            self.tile_variant = 0
                        if event.button == 5:
                            self.tile_group = (self.tile_group + 1) % len(
                                self.tile_list
                            )
                            self.tile_variant = 0

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.left_clicking = False
                    if event.button == 3:
                        self.right_clicking = False

            self.scroll[0] += 2 * (self.movement[1] - self.movement[0])
            self.scroll[1] += 2 * (self.movement[3] - self.movement[2])
            render_scroll = (int(self.scroll[0]), int(self.scroll[1]))

            # Add tiles
            if self.left_clicking and self.on_grid:
                self.tile_map.tile_map[f"{tile_pos[0]};{tile_pos[1]}"] = {
                    "type": self.tile_list[self.tile_group],
                    "variant": self.tile_variant,
                    "pos": tile_pos,
                }

            if self.right_clicking:
                tile_loc = f"{tile_pos[0]};{tile_pos[1]}"
                if tile_loc in self.tile_map.tile_map:
                    del self.tile_map.tile_map[tile_loc]
                # Check offgrid tiles
                for tile in self.tile_map.offgrid_tiles.copy():
                    tile_img = self.assets[tile["type"]][tile["variant"]]
                    tile_rect = pygame.Rect(
                        tile["pos"][0] - self.scroll[0],
                        tile["pos"][1] - self.scroll[1],
                        tile_img.get_width(),
                        tile_img.get_height(),
                    )
                    if tile_rect.collidepoint(mpos):
                        self.tile_map.offgrid_tiles.remove(tile)

            # Update selected tile
            current_tile_img = self.assets[self.tile_list[self.tile_group]][
                self.tile_variant
            ].copy()
            current_tile_img.set_alpha(100)

            # Draw
            self.display.blit(self.assets["background"], (0, 0))
            self.display.blit(current_tile_img, (5, 5))
            if self.on_grid:
                self.display.blit(
                    current_tile_img,
                    (
                        tile_pos[0] * self.tile_map.tile_size - self.scroll[0],
                        tile_pos[1] * self.tile_map.tile_size - self.scroll[1],
                    ),
                )
            else:
                self.display.blit(current_tile_img, mpos)
            self.tile_map.draw(self.display, offset=render_scroll)
            self.window.blit(
                pygame.transform.scale_by(self.display, RENDER_SCALE), (0, 0)
            )
            pygame.display.update()


def main() -> None:
    Editor().run()


if __name__ == "__main__":
    main()

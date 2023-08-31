import pygame


class Tile:
    def __init__(self, x: int, y: int, tile_size: int, img: pygame.surface.Surface, window: pygame.surface.Surface) -> None:
        self.rect = pygame.Rect((x, y, tile_size, tile_size))
        self.image = pygame.transform.scale(img, (tile_size, tile_size))
        self.window = window

    def draw(self) -> None:
        self.window.blit(self.image, (self.rect.x, self.rect.y))

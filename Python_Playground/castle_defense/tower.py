import pygame


class Tower(pygame.sprite.Sprite):
    def __init__(self, sprites: dict[str, pygame.surface.Surface], x: int, y: int) -> None:
        pygame.sprite.Sprite.__init__(self)
        self.sprites = sprites
        self.image = self.sprites['tower_100']
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.bullet_start = self.rect.center

    def set_25(self) -> None:
        self.image = self.sprites['tower_25']

    def set_50(self) -> None:
        self.image = self.sprites['tower_50']

    def set_100(self) -> None:
        self.image = self.sprites['tower_100']

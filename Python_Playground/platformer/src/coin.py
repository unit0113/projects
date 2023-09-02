import pygame


class Coin(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int, img: pygame.surface.Surface) -> None:
        pygame.sprite.Sprite.__init__(self)
        self.image = img
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y

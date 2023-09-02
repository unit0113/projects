import pygame


class Gate(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int, img: pygame.surface.Surface) -> None:
        pygame.sprite.Sprite.__init__(self)
        self.image = img
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

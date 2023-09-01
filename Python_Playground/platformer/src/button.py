import pygame


class Button:
    def __init__(self, x: int, y: int, img: pygame.surface.Surface, window: pygame.surface.Surface) -> None:
        self.image = img
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.window = window

    def draw(self) -> None:
        self.window.blit(self.image, (self.rect.x, self.rect.y))

    def is_clicked(self, pos: tuple[int, int]):
        return self.rect.collidepoint(pos)
    
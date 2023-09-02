import pygame


MAX_MOVE_DIST = 50


class Platform(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int, img: pygame.surface.Surface, move_x: bool=True) -> None:
        pygame.sprite.Sprite.__init__(self)
        self.image = img
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.move_x = move_x
        self.move_direction = 1
        self.move_counter = 0

    def update(self) -> None:
        if self.move_x:
            self.rect.x += self.move_direction
        else:
            self.rect.y += self.move_direction
        self.move_counter += self.move_direction
        if abs(self.move_counter) > MAX_MOVE_DIST:
            self.move_direction *= -1
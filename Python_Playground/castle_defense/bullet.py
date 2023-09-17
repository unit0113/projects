import pygame
import math

from settings import WIDTH, HEIGHT


class Bullet(pygame.sprite.Sprite):
    def __init__(self, start_x: int, start_y: int, target_x: int, target_y: int, speed: int, img: pygame.surface.Surface) -> None:
        pygame.sprite.Sprite.__init__(self)
        self.image = img
        self.rect = self.image.get_rect()
        x , y = start_x - self.image.get_width() // 2, start_y - self.image.get_height() // 2
        self.rect.center = (x, y)
        self.start_x = x
        self.start_y = y
        angle = math.atan2(-(target_y - start_y), (target_x - start_x))
        self.dx = math.cos(angle) * speed
        self.dy = -math.sin(angle) * speed
        self.timer = 0

    def update(self, dt) -> None:
        self.timer += dt
        self.rect.x = int(self.timer * self.dx + self.start_x)
        self.rect.y = int(self.timer * self.dy + self.start_y)

        # Check if off screen
        if self.is_offscreen():
            self.kill()

    def is_offscreen(self) -> bool:
        return (self.rect.right < 0
                or self.rect.left > WIDTH
                or self.rect.bottom < 0
                or self.rect.top > HEIGHT)

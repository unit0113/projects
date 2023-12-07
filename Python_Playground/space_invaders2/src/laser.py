import pygame
import math


class Laser(pygame.sprite.Sprite):
    def __init__(
        self,
        x: int,
        y: int,
        color: str,
        shape: str,
        speed: int,
        direction: tuple[float, float],
    ) -> None:
        pygame.sprite.Sprite.__init__(self)
        self.speed = speed
        self.direction = pygame.math.Vector2(direction)

        # Load image
        image = pygame.image.load(
            f"src/assets/projectiles/laser{color}{shape}.png"
        ).convert_alpha()
        angle = math.degrees(math.atan2(-self.direction[1], self.direction[0]))
        self.image = pygame.transform.rotate(image, angle - 90)
        self.rect = self.image.get_rect()
        self.rect.midtop = (x, y)

    def update(self, dt: float) -> None:
        self.rect.center += self.speed * self.direction * dt

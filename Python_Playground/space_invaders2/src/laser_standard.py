import pygame
import math
import random


class StandardLaser(pygame.sprite.Sprite):
    def __init__(
        self,
        pos: tuple[int, int],
        image: pygame.Surface,
        speed: int,
        direction: tuple[float, float],
        damage: float,
        dispersion: float,
    ) -> None:
        pygame.sprite.Sprite.__init__(self)

        self.speed = speed
        self.direction = pygame.math.Vector2(direction).rotate(
            random.randrange(-dispersion, dispersion)
        )
        self.damage = damage

        # Create image
        angle = math.degrees(math.atan2(-self.direction[1], self.direction[0]))
        self.image = pygame.transform.rotate(image, angle - 90)
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.midtop = pos

    def update(self, dt: float) -> None:
        """Update game objects in game loop

        Args:
            dt (float): time since last frame
        """

        self.rect.center += self.speed * self.direction * dt

    def get_damage(self) -> float:
        return self.damage

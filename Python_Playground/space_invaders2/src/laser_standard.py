import pygame
import math


class StandardLaser(pygame.sprite.Sprite):
    def __init__(
        self,
        pos: tuple[int, int],
        image: pygame.Surface,
        speed: int,
        direction: tuple[float, float],
    ) -> None:
        pygame.sprite.Sprite.__init__(self)

        self.speed = speed
        self.direction = pygame.math.Vector2(direction)

        # Create image
        angle = math.degrees(math.atan2(-self.direction[1], self.direction[0]))
        self.image = pygame.transform.rotate(image, angle - 90)
        self.rect = self.image.get_rect()
        self.rect.midtop = pos

    def update(self, dt: float) -> None:
        """Update game objects in game loop

        Args:
            dt (float): time since last frame
        """

        self.rect.center += self.speed * self.direction * dt

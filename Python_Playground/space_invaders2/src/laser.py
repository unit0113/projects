import pygame
import math
import random


class Laser(pygame.sprite.Sprite):
    def __init__(
        self,
        pos: tuple[int, int],
        image: pygame.Surface,
        speed: int,
        direction: tuple[float, float],
        damages: tuple[float, float],
        dispersion: float,
    ) -> None:
        pygame.sprite.Sprite.__init__(self)

        self.speed = speed
        self.direction = pygame.math.Vector2(direction)
        if dispersion:
            self.direction = self.direction.rotate(
                random.randrange(-dispersion, dispersion)
            )
        self.shield_damage, self.ship_damage = damages

        # Create image
        angle = math.degrees(math.atan2(-self.direction[1], self.direction[0]))
        self.image = pygame.transform.rotate(image, angle - 90)
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.pos = pos
        self.rect.midtop = pos

    def update(self, dt: float, enemies: pygame.sprite.Group = None) -> None:
        """Update game object in game loop

        Args:
            dt (float): time since last frame
            enemies (pygame.sprite.Group): enemies currently in game, unused
        """

        self.pos += self.speed * self.direction * dt
        self.rect.center = self.pos

    def get_shield_damage(self) -> float:
        """Getter for shield damage

        Returns:
            float: damage done to shields
        """
        self.kill()
        return self.shield_damage

    def get_ship_damage(self) -> float:
        """Getter for ship damage

        Returns:
            float: damage done to ships
        """
        self.kill()
        return self.ship_damage

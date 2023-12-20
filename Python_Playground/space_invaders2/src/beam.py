import pygame
import math


class Beam(pygame.sprite.Sprite):
    def __init__(
        self,
        pos: tuple[int, int],
        image: pygame.Surface,
        direction: tuple[float, float],
        damages: tuple[float, float],
        width: float,
    ) -> None:
        pygame.sprite.Sprite.__init__(self)

        self.direction = pygame.math.Vector2(direction)
        self.shield_damage, self.ship_damage = damages
        self.width = width

        # Create image
        angle = math.degrees(math.atan2(-self.direction[1], self.direction[0]))
        self.image = pygame.transform.rotate(image, angle - 90)
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        if direction[1] > 0:
            self.rect.midtop = pos
        else:
            self.rect.midbottom = pos

    def update(self, dt: float, enemies: pygame.sprite.Group = None) -> None:
        """Update game object in game loop

        Args:
            dt (float): time since last frame
            enemies (pygame.sprite.Group): enemies currently in game, unused
        """

        self.kill()

    def get_shield_damage(self) -> float:
        """Getter for shield damage

        Returns:
            float: damage done to shields
        """
        return self.shield_damage

    def get_ship_damage(self) -> float:
        """Getter for ship damage

        Returns:
            float: damage done to ships
        """
        return self.ship_damage

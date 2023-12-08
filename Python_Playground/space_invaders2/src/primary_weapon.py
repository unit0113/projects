import pygame
from abc import ABC, abstractmethod


class PrimaryWeapon(ABC):
    def __init__(
        self, muzzle_pos_offset: tuple[int, int], cooldown: int, projectile_type: str
    ) -> None:
        self.muzzle_pos_offset = muzzle_pos_offset
        self.cooldown = cooldown
        self.last_shot = pygame.time.get_ticks()

        self.projectile_image = pygame.image.load(
            f"src/assets/projectiles/laser{projectile_type}.png"
        ).convert_alpha()
        self.projectile_image = pygame.transform.scale_by(self.projectile_image, 0.5)

    @abstractmethod
    def fire(self, muzzle_pos: tuple[int, int]):
        ...

    @property
    def _can_fire(self) -> bool:
        return pygame.time.get_ticks() > self.last_shot + self.cooldown

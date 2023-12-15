import pygame
from abc import ABC, abstractmethod


class PrimaryWeapon(ABC):
    def __init__(
        self,
        muzzle_pos_offset: tuple[int, int],
        cooldown: int,
        projectile_type: str,
        base_dmg: int,
        muzzle_velocity: int,
        dispersion: float,
    ) -> None:
        self.muzzle_pos_offset = muzzle_pos_offset
        self.base_cooldown = cooldown
        self.cooldown = cooldown
        self.base_dmg = base_dmg
        self.dmg = base_dmg
        self.muzzle_velocity = muzzle_velocity
        self.dispersion = dispersion
        self.last_shot = pygame.time.get_ticks()

        self.projectile_image = pygame.image.load(
            f"src/assets/projectiles/laser{projectile_type}.png"
        ).convert_alpha()
        self.projectile_image = pygame.transform.scale_by(self.projectile_image, 0.5)

    @abstractmethod
    def fire(self, muzzle_pos: tuple[int, int]):
        """Returns the projectiles that the ship fired this frame

        Returns:
            Optional[Projectile]: projectiles fired
        """

        ...

    @abstractmethod
    def upgrade(self) -> None:
        """Upgrade weapon stats"""
        ...

    def set_upgrade_level(self, level) -> None:
        self.dmg = self.base_dmg
        self.cooldown = self.base_cooldown

        for _ in range(level - 1):
            self.upgrade()

    @property
    def _can_fire(self) -> bool:
        """Determines whether the weapon is capable of firing

        Returns:
            bool: if weapon can fire
        """

        return pygame.time.get_ticks() > self.last_shot + self.cooldown

import pygame
from typing import Optional

from .laser_standard import StandardLaser


class LaserCannon:
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
        self.level = 1
        self.last_shot = pygame.time.get_ticks()

        self.projectile_image = pygame.image.load(
            f"src/assets/projectiles/laser{projectile_type}.png"
        ).convert_alpha()
        self.projectile_image = pygame.transform.scale_by(self.projectile_image, 0.5)

    def fire(
        self, ship_pos: tuple[int, int], direction: tuple[int, int]
    ) -> Optional[StandardLaser]:
        """Returns the projectiles that the ship fired this frame

        Returns:
            Optional[Projectile]: projectiles fired
        """

        if self._can_fire:
            self.last_shot = pygame.time.get_ticks()
            return StandardLaser(
                (
                    ship_pos[0] + self.muzzle_pos_offset[0],
                    ship_pos[1] + self.muzzle_pos_offset[1],
                ),
                self.projectile_image,
                self.muzzle_velocity,
                direction,
                self.dmg,
                self.dispersion,
            )

    @property
    def _can_fire(self) -> bool:
        """Determines whether the weapon is capable of firing

        Returns:
            bool: if weapon can fire
        """

        return pygame.time.get_ticks() > self.last_shot + self.cooldown

    def upgrade(self) -> None:
        self.cooldown *= 0.95
        self.dmg *= 1.05
        self.level += 1

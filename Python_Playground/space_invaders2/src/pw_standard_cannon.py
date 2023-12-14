import pygame
from typing import Optional

from .primary_weapon import PrimaryWeapon
from .laser_standard import StandardLaser


class StandardCannon(PrimaryWeapon):
    def __init__(
        self,
        muzzle_pos_offset: tuple[int, int],
        cooldown: int,
        projectile_type: str,
        base_dmg: int,
        muzzle_velocity: int,
        dispersion: float,
    ) -> None:
        super().__init__(
            muzzle_pos_offset,
            cooldown,
            projectile_type,
            base_dmg,
            muzzle_velocity,
            dispersion,
        )

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

    def upgrade(self) -> None:
        self.cooldown *= 0.95
        self.dmg *= 1.05

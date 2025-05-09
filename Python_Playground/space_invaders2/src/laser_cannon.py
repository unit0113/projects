import pygame
from typing import Optional

from .laser import Laser


class LaserCannon:
    def __init__(
        self,
        muzzle_pos_offset: tuple[int, int],
        cooldown: float,
        projectile_image: pygame.Surface,
        base_damages: tuple[float, float],
        muzzle_velocity: float,
        dispersion: float,
        direction: tuple[float, float],
    ) -> None:
        self.muzzle_pos_offset = muzzle_pos_offset
        self.base_cooldown = cooldown
        self.cooldown = cooldown
        self.base_damages = base_damages
        self.damages = base_damages
        self.muzzle_velocity = muzzle_velocity
        self.dispersion = dispersion
        self.direction = direction
        self.last_shot = pygame.time.get_ticks()

        self.projectile_image = projectile_image

    def fire(self, ship_pos: tuple[int, int]) -> Optional[Laser]:
        """Returns the projectiles that the ship fired this frame

        Returns:
            Optional[Projectile]: projectiles fired
        """

        if self._can_fire:
            self.last_shot = pygame.time.get_ticks()
            return Laser(
                (
                    ship_pos[0] + self.muzzle_pos_offset[0],
                    ship_pos[1] + self.muzzle_pos_offset[1],
                ),
                self.projectile_image,
                self.muzzle_velocity,
                self.direction,
                self.damages,
                self.dispersion,
            )

    @property
    def _can_fire(self) -> bool:
        """Determines whether the weapon is capable of firing

        Returns:
            bool: if weapon can fire
        """

        return pygame.time.get_ticks() > self.last_shot + self.cooldown

    def get_status(self) -> float:
        return (pygame.time.get_ticks() - self.last_shot) / self.cooldown

import pygame
from typing import Optional

from .beam import Beam
from .settings import HEIGHT


class BeamWeapon:
    def __init__(
        self,
        muzzle_pos_offset: tuple[int, int],
        cooldown: float,
        duration: float,
        projectile_color: str,
        base_damages: tuple[float, float],
        width: float,
        direction: tuple[int, int],
    ) -> None:
        self.muzzle_pos_offset = muzzle_pos_offset
        self.base_cooldown = cooldown
        self.cooldown = cooldown
        self.base_duration = duration
        self.duration = duration
        self.base_damages = base_damages
        self.damages = base_damages
        self.width = width
        self.direction = direction
        self.last_shot = pygame.time.get_ticks()
        self.start_of_shot = pygame.time.get_ticks()

        self.projectile_image = pygame.image.load(
            f"src/assets/projectiles/beam{projectile_color}.png"
        ).convert_alpha()
        self.projectile_image = pygame.transform.scale(
            self.projectile_image, (self.width, HEIGHT)
        )

    def fire(self, ship_pos: tuple[int, int]) -> Optional[Beam]:
        """Returns the projectiles that the ship fired this frame

        Returns:
            Optional[Projectile]: projectiles fired
        """

        if self._can_fire:
            return Beam(
                (
                    ship_pos[0] + self.muzzle_pos_offset[0],
                    ship_pos[1] + self.muzzle_pos_offset[1],
                ),
                self.projectile_image,
                self.direction,
                self.damages,
                self.width,
            )

    @property
    def _can_fire(self) -> bool:
        """Determines whether the weapon is capable of firing

        Returns:
            bool: if weapon can fire
        """

        if pygame.time.get_ticks() > self.last_shot + self.cooldown:
            self.start_of_shot = self.last_shot = pygame.time.get_ticks()
            return True
        elif pygame.time.get_ticks() < self.start_of_shot + self.duration:
            self.last_shot = pygame.time.get_ticks()
            return True
        return False

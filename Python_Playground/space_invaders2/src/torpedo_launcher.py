import pygame

from .torpedo import Torpedo


class TorpedoLauncher:
    launcher_is_left = [True, False]

    def __init__(
        self,
        offsets: list[tuple[int, int]],
        cooldown: float,
        base_damages: tuple[float, float],
        missile_speed: int,
        direction: tuple[float, float],
    ) -> None:
        self.base_damages = base_damages
        self.damages = base_damages
        self.base_cooldown = cooldown
        self.cooldown = cooldown
        self.offsets = offsets
        self.direction = direction
        self.missile_speed = missile_speed
        self.last_shot = 0
        self.load_sprites()

    def load_sprites(self) -> None:
        sprite_sheet = pygame.image.load(
            f"src/assets/projectiles/torpedo.png"
        ).convert_alpha()
        num_cols = 3
        size_h = sprite_sheet.get_width() // num_cols
        size_v = sprite_sheet.get_height()

        self.sprites = []

        for column in range(num_cols):
            sprite = sprite_sheet.subsurface(column * size_h, 0, size_h, size_v)
            # If owner is enemy
            if self.direction == (0, 1):
                sprite = pygame.transform.rotate(sprite, 180)
            self.sprites.append(sprite)

    def fire(self, ship_pos: tuple[int, int]):
        if self._can_fire:
            self.last_shot = pygame.time.get_ticks()
            return [
                Torpedo(
                    self.sprites,
                    (ship_pos[0] + offset[0], ship_pos[1] + offset[1]),
                    self.damages,
                    self.missile_speed,
                    self.direction,
                    is_left,
                )
                for offset, is_left in zip(
                    self.offsets, TorpedoLauncher.launcher_is_left
                )
            ]

    @property
    def _can_fire(self) -> bool:
        """Determines whether the weapon is capable of firing

        Returns:
            bool: if weapon can fire
        """

        return pygame.time.get_ticks() > self.last_shot + self.cooldown

    def get_status(self) -> float:
        return (pygame.time.get_ticks() - self.last_shot) / self.cooldown

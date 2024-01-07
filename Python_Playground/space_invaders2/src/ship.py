import pygame
from abc import ABC, abstractmethod

from .weapon_factories import PrimaryWeaponFactory, SecondaryWeaponFactory
from .shield import Shield


class Ship(ABC):
    def __init__(self) -> None:
        self.shield = None
        self.last_hit = 0

    def add_shield(self, size_ratio: float) -> None:
        """Adds shield to ship

        Args:
            size_ratio (float): Amount to divide image size by for shield size
        """

        self.shield = Shield(
            self.base_shield_strength,
            self.base_shield_regen,
            self.base_shield_cooldown,
            self.image.get_width() // size_ratio,
        )

    @abstractmethod
    def update(self, dt: float) -> None:
        """Update game objects in game loop

        Args:
            dt (float): time since last frame
        """

        pass

    @abstractmethod
    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        pass

    def split_sprite_sheet(
        self,
        sprite_sheet: pygame.Surface,
        num_rows: int,
        num_cols: int,
        *,
        scale: float = 1
    ) -> dict[str, list[pygame.Surface]]:
        """Parses the spritesheet of the provided ship_type

        Args:
            sprite_sheet (pygame.Surface): Sprite sheet to parse
            num_rows (int): Number of rows
            num_cols (int): Number of columns
            scale (float, optional): Factor to scale sprites by. Defaults to 1.

        Returns:
            dict[str, list[pygame.Surface]]: dict containing animation frames
        """

        size_h = sprite_sheet.get_width() // num_cols
        size_v = sprite_sheet.get_height() // num_rows

        orientations = ["level", "left", "full_left", "right", "full_right"]
        sprites = {}

        for row, orientation in zip(range(num_rows), orientations):
            sprites[orientation] = []
            for column in range(num_cols):
                sprite = sprite_sheet.subsurface(
                    column * size_h, row * size_v, size_h, size_v
                )
                if scale > 1:
                    sprite = pygame.transform.scale_by(sprite, scale)
                sprites[orientation].append(sprite)

        return sprites

    def load_weapons(self, ship_data: dict, is_player: bool = False) -> None:
        """Parse and create the weapons as specified in the ship_data

        Args:
            ship_data (dict): specifications for the ship
        """
        primary_weapon_factory = PrimaryWeaponFactory()
        self.primary_weapons = []
        for weapon, offset in ship_data["primary_weapons"]:
            self.primary_weapons.append(
                primary_weapon_factory.get_weapon(
                    weapon, offset, self.projectile_color, is_player
                )
            )

        secondary_weapon_factory = SecondaryWeaponFactory()
        self.secondary_weapons = []
        for weapon, offsets in zip(
            ship_data["secondary_weapons"], self.secondary_offsets
        ):
            self.secondary_weapons.append(
                secondary_weapon_factory.get_weapon(
                    weapon, offsets, self.projectile_color, is_player
                )
            )

    def take_damage(self, damage: float) -> None:
        """Take damage to ship

        Args:
            damage (float): amount of damage
        """

        self.health -= damage
        self.last_hit = pygame.time.get_ticks()

    @property
    def is_dead(self) -> bool:
        """Whether ship is dead

        Returns:
            bool: if ship is dead
        """

        return self.health <= 0

    @property
    def shield_active(self) -> bool:
        """Whether shield is active

        Returns:
            bool: if shield is active
        """

        return self.shield and self.shield.active

    def get_status(self) -> tuple[float, float, float, float]:
        """Returns the status of ship systems for display in UI

        Returns:
            tuple[float, float, float, float]: statuses of health, shield, primary weapons and secondary weapons
        """

        health_status = self.health / self.max_health
        shield_status = None if not self.shield else self.shield.get_status()
        primary_weapon_status = (
            None
            if not self.primary_weapons
            else min(1, self.primary_weapons[0].get_status())
        )
        secondary_weapon_statuses = (
            None
            if not self.secondary_weapons
            else [min(1, weapon.get_status()) for weapon in self.secondary_weapons]
        )

        return (
            health_status,
            shield_status,
            primary_weapon_status,
            secondary_weapon_statuses,
        )

    def get_rect_data(self) -> tuple[int, int, int, int]:
        """Returns data on ship rect for placement of UI elements

        Returns:
            tuple[int, int, int, int]: x, y, width, height of rect object
        """

        return self.rect.x, self.rect.y, self.rect.width, self.rect.height

    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        window.blit(self.image, self.rect)
        if self.shield:
            self.shield.draw(window)

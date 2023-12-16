import pygame
from abc import ABC, abstractmethod

from .weapon_factories import PrimaryWeaponFactory, SecondaryWeaponFactory


class Ship(ABC):
    def __init__(self, ship_data: dict) -> None:
        self.health = ship_data["hp"]
        self.max_health = self.health
        self.speed = ship_data["speed"]
        self.secondary_offsets = ship_data["secondary_offsets"]
        self.projectile_color = ship_data["projectile_color"]
        self.shield = None
        self.last_hit = 0

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

    def load_sprite_sheet(
        self, ship_type: str, num_rows: int, num_cols: int
    ) -> dict[str, list[pygame.Surface]]:
        """Loads and parses the spritesheet of the provided ship_type

        Args:
            ship_type (str): name of ship to load

        Returns:
            dict[str, list[pygame.Surface]]: dict containing animation frames
        """

        sprite_sheet = pygame.image.load(
            f"src/assets/ships/{ship_type}.png"
        ).convert_alpha()
        size_h = sprite_sheet.get_width() // num_cols
        size_v = sprite_sheet.get_height() // num_rows

        orientations = ["level", "left", "full_left", "right", "full_right"]
        sprites = {}

        for row, orientation in zip(range(num_rows), orientations):
            sprites[orientation] = []
            for column in range(num_cols):
                sprites[orientation].append(
                    sprite_sheet.subsurface(
                        column * size_h, row * size_v, size_h, size_v
                    )
                )

        return sprites

    def load_weapons(self, ship_data: dict, is_player: bool = False) -> None:
        """Parse and create the weapons as specified in the ship_data

        Args:
            ship_data (dict): specifications for the ship
        """

        self.primary_weapons = []
        for weapon, offset in ship_data["primary_weapons"]:
            self.primary_weapons.append(
                PrimaryWeaponFactory.get_weapon(
                    weapon, offset, self.projectile_color, is_player
                )
            )

        self.secondary_weapons = []
        for weapon in ship_data["secondary_weapons"]:
            self.secondary_weapons.extend(
                SecondaryWeaponFactory.get_weapon(
                    weapon, self.secondary_offsets, self.projectile_color, is_player
                )
            )

    def take_damage(self, damage: float) -> None:
        self.health -= damage
        self.last_hit = pygame.time.get_ticks()

    @property
    def is_dead(self) -> bool:
        return self.health <= 0

    @property
    def shield_active(self) -> bool:
        return self.shield and self.shield.active

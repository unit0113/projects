import pygame
from abc import ABC, abstractmethod


class Ship(ABC):
    def __init__(self) -> None:
        pass

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

    def load_weapons(self, ship_data: dict) -> None:
        """Parse and create the weapons as specified in the ship_data

        Args:
            ship_data (dict): specifications for the ship
        """

        self.primary_weapons = []
        for weapon, args in ship_data["primary_weapons"]:
            self.primary_weapons.append(weapon(*args))

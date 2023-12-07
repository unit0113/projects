import pygame
from abc import ABC, abstractmethod


class Ship(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def update(self, dt: float) -> None:
        pass

    @abstractmethod
    def draw(self, window: pygame.Surface) -> None:
        pass

    def load_sprite_sheet(self, ship_type: str) -> dict[str, list[pygame.Surface]]:
        sprite_sheet = pygame.image.load(
            f"src/assets/ships/{ship_type}.png"
        ).convert_alpha()
        size_h = sprite_sheet.get_width() // 3
        size_v = sprite_sheet.get_height() // 3
        orientations = ["level", "right", "left"]
        sprites = {}

        for row, orientation in enumerate(orientations):
            sprites[orientation] = []
            for column in range(3):
                sprites[orientation].append(
                    sprite_sheet.subsurface(
                        column * size_h, row * size_v, size_h, size_v
                    )
                )

        return sprites

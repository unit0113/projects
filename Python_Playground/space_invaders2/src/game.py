import pygame

from .state_stack import StateStack
from .background import Background
from .player_ship import PlayerShip
from .laser import Laser


class Game:
    def __init__(self) -> None:
        self.load_assets()
        self.background = Background(
            self.assets["background"], self.assets["foreground"]
        )
        self.state_stack = StateStack()

        self.player = PlayerShip("boomerang")
        self.laserGroup = pygame.sprite.Group()
        self.laserGroup.add(Laser(400, 100, "Green", "ThinLong", 500, (1, 1)))

    def load_assets(self) -> None:
        self.assets = {}
        path = "src/assets"

        # Load background
        folder = path + "/background"
        self.assets["background"] = pygame.image.load(
            f"{folder}/nebula_b.png"
        ).convert_alpha()
        self.assets["foreground"] = pygame.image.load(
            f"{folder}/stars.png"
        ).convert_alpha()

        # Load lasers

    def update(self, dt: float) -> None:
        self.background.update(dt)
        self.player.update(dt)
        self.laserGroup.update(dt)

    def draw(self, window: pygame.Surface) -> None:
        window.fill((0, 0, 0))
        self.background.draw(window)
        self.player.draw(window)
        self.laserGroup.draw(window)

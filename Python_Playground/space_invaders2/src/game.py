import pygame

from .state_stack import StateStack
from .background import Background
from .state_test import TestState
from .settings import WIDTH, HEIGHT
from .player_ship import PlayerShip


class Game:
    def __init__(self) -> None:
        self.load_assets()
        self.background = Background(
            self.assets["background"], self.assets["foreground"]
        )

        self.state_stack = StateStack()
        self.state_stack.push(TestState(self))

        self.playerLaserGroup = pygame.sprite.Group()

    def set_player(self, player: PlayerShip) -> None:
        self.player = player

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

    def update(self, dt: float) -> None:
        self.background.update(dt)
        self.state_stack.update(dt)

    def update_player(self, dt: float) -> None:
        self.player.update(dt)

    def update_projectiles(self, dt: float) -> None:
        self.playerLaserGroup.update(dt)

    def fire(self) -> None:
        player_projectiles = self.player.fire()
        if player_projectiles:
            self.playerLaserGroup.add(player_projectiles)

    def remove_offscreen_objects(self) -> None:
        for laser in self.playerLaserGroup:
            if not self.object_is_onscreen(laser):
                laser.kill()

    def object_is_onscreen(self, obj) -> bool:
        return (0 - obj.rect.width <= obj.rect.x <= WIDTH) and (
            0 - obj.rect.height <= obj.rect.y <= HEIGHT
        )

    def draw(self, window: pygame.Surface) -> None:
        window.fill((0, 0, 0))
        self.background.draw(window)
        self.state_stack.draw(window)

    def draw_player(self, window: pygame.Surface) -> None:
        self.player.draw(window)

    def draw_projectiles(self, window: pygame.Surface) -> None:
        self.playerLaserGroup.draw(window)

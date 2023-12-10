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

        # Initialize game asset groups
        self.playerLaserGroup = pygame.sprite.Group()

    def set_player(self, player: PlayerShip) -> None:
        """Recieves the player ship from the ship select state

        Args:
            player (PlayerShip): the players ship
        """

        self.player = player

    def load_assets(self) -> None:
        """Loads and stores game assets"""

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
        """Update game objects in game loop

        Args:
            dt (float): time since last frame
        """

        self.background.update(dt)
        self.state_stack.update(dt)

    def update_player(self, dt: float) -> None:
        """Public method to allow states to update player ship

        Args:
            dt (float): time since last frame
        """

        self.player.update(dt)

    def update_projectiles(self, dt: float) -> None:
        """Public method to allow states to update projectile groups

        Args:
            dt (float): time since last frame
        """

        self.playerLaserGroup.update(dt)

    def fire(self) -> None:
        """Public method to allow ships to fire and add the
        resultant projectiles to the appropriate group
        """

        player_projectiles = self.player.fire()
        if player_projectiles:
            self.playerLaserGroup.add(player_projectiles)

    def remove_offscreen_objects(self) -> None:
        """Deletes objects that have fallen off the screen"""

        for laser in self.playerLaserGroup:
            if not self.object_is_onscreen(laser):
                laser.kill()

    def object_is_onscreen(self, obj) -> bool:
        """Determines whether provide object is off screen

        Args:
            obj (_type_): game object to check, must have .rect attribute

        Returns:
            bool: _description_
        """

        return (0 - obj.rect.width <= obj.rect.x <= WIDTH) and (
            0 - obj.rect.height <= obj.rect.y <= HEIGHT
        )

    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        window.fill((0, 0, 0))
        self.background.draw(window)
        self.state_stack.draw(window)

    def draw_player(self, window: pygame.Surface) -> None:
        """Public method to allow states to draw player

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        self.player.draw(window)

    def draw_projectiles(self, window: pygame.Surface) -> None:
        """Public method to allow states to draw projectiles

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        self.playerLaserGroup.draw(window)

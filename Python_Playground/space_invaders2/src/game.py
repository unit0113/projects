import pygame
from typing import Callable

from .state_stack import StateStack
from .background import Background
from .state_test import TestState
from .settings import WIDTH, HEIGHT
from .player_ship import PlayerShip
from .enemy import Enemy


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
        self.enemyGroup = pygame.sprite.Group()
        self.enemyLaserGroup = pygame.sprite.Group()
        self.enemyGroup.add(Enemy("bug_3_b", WIDTH // 2, 0))

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

    def update_enemies(self, dt: float) -> None:
        """Public method to allow states to update enemies groups

        Args:
            dt (float): time since last frame
        """

        self.enemyGroup.update(dt)

    def update_projectiles(self, dt: float) -> None:
        """Public method to allow states to update projectile groups

        Args:
            dt (float): time since last frame
        """

        self.playerLaserGroup.update(dt)
        self.enemyLaserGroup.update(dt)

    def check_collisions(self) -> None:
        """Public method to determine collisions and calculate damage done"""

        # Check damage to enemies
        for enemy in self.enemyGroup:
            for laser in self.playerLaserGroup:
                if enemy.shield_active and pygame.sprite.collide_circle(
                    enemy.shield, laser
                ):
                    enemy.shield.take_damage(laser.get_damage())
                    laser.kill()
                elif self.is_collision(enemy, laser):
                    enemy.take_damage(laser.get_damage())
                    laser.kill()
                    if enemy.is_dead:
                        break
            if enemy.is_dead:
                pass

        # Check damage to player
        for laser in self.enemyLaserGroup:
            if self.player.shield_active and pygame.sprite.collide_circle(
                self.player.shield, laser
            ):
                self.player.shield.take_damage(laser.get_damage())
                laser.kill()
            elif self.is_collision(self.player, laser):
                self.player.take_damage(laser.get_damage())
                laser.kill()

        # Check ship to ship collision
        for enemy in self.enemyGroup:
            if self.is_collision(enemy, self.player):
                self.player.take_damage(enemy.max_health)
                enemy.kill()

    def is_collision(self, obj1: object, obj2: object) -> bool:
        """Effeciently check for object collisions

        Args:
            obj1 (object): game object
            obj2 (object): game object

        Returns:
            bool: if objects are colliding
        """
        return pygame.Rect.colliderect(obj1.rect, obj2.rect) and obj1.mask.overlap(
            obj2.mask, (obj2.rect.x - obj1.rect.x, obj2.rect.y - obj1.rect.y)
        )

    def fire(self) -> None:
        """Public method to allow ships to fire and add the
        resultant projectiles to the appropriate group
        """

        # Get player projectiles
        player_projectiles = self.player.fire()
        if player_projectiles:
            self.playerLaserGroup.add(player_projectiles)

        # Get enemy projectiles
        for enemy in self.enemyGroup:
            enemy_projectiles = enemy.fire()
            if enemy_projectiles:
                self.enemyLaserGroup.add(enemy_projectiles)

    def remove_offscreen_objects(self) -> None:
        """Deletes objects that have fallen off the screen"""

        for laser in self.playerLaserGroup:
            if not self.object_is_onscreen(laser, [self.is_offscreen_up]):
                laser.kill()

        for laser in self.enemyLaserGroup:
            if not self.object_is_onscreen(laser, [self.is_offscreen_down]):
                laser.kill()

        for enemy in self.enemyGroup:
            if not self.object_is_onscreen(enemy, [self.is_offscreen_down]):
                enemy.kill()

    def object_is_onscreen(self, obj: object, conditionals: list[Callable]) -> bool:
        """Determines whether provide object is off screen in specified directions

        Args:
            obj (_type_): game object to check, must have .rect attribute

        Returns:
            bool: _description_
        """

        return all([not conditional(obj) for conditional in conditionals])

    def is_offscreen_up(self, obj: object) -> bool:
        return 0 > obj.rect.y + obj.rect.height

    def is_offscreen_down(self, obj: object) -> bool:
        return obj.rect.y > HEIGHT

    def is_offscreen_left(self, obj: object) -> bool:
        return 0 > obj.rect.x + obj.rect.width

    def is_offscreen_right(self, obj: object) -> bool:
        return obj.rect.x > WIDTH

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

    def draw_enemies(self, window: pygame.Surface) -> None:
        """Public method to allow states to draw enemies

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        for enemy in self.enemyGroup:
            enemy.draw(window)

    def draw_projectiles(self, window: pygame.Surface) -> None:
        """Public method to allow states to draw projectiles

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        self.playerLaserGroup.draw(window)
        self.enemyLaserGroup.draw(window)

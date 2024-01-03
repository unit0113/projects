import pygame
from typing import Callable

from .state_stack import StateStack
from .background import Background
from .state_test import TestState
from .state_menu import MenuState
from .settings import WIDTH, HEIGHT, YELLOW, RED, GREEN, SHIELD_BLUE, GREY, NUM_LIVES
from .player_ship import PlayerShip
from .level_generator import LevelGenerator

DEV_MODE = False


class Game:
    def __init__(self) -> None:
        self.load_assets()
        self.background = Background(
            self.assets["background"], self.assets["foreground"]
        )

        # Initialize game asset groups
        self.playerProjectileGroup = pygame.sprite.Group()
        self.enemy_group = pygame.sprite.Group()
        self.enemyProjectileGroup = pygame.sprite.Group()
        self.player_group = pygame.sprite.Group()

        # Initialize state stack
        self.state_stack = StateStack()

        # Initialize level generator
        self.level_generator = LevelGenerator()
        self.level_generator.generate_level()

        self.reset()

    def reset(self) -> None:
        """Reset the game"""
        # Empty groups
        self.playerProjectileGroup.empty()
        self.enemy_group.empty()
        self.enemyProjectileGroup.empty()
        self.player_group.empty()

        # Clear state stack
        self.state_stack.empty()

        # Reset game variables
        self.score = 0
        self.money = 0
        self.level = 1
        self.lives = NUM_LIVES
        self.num_bombs = 3

        # Start first state
        self.level_generator.generate_level()
        if DEV_MODE:
            self.state_stack.push(TestState(self))
        else:
            self.state_stack.push(MenuState(self))

    def next_level(self) -> None:
        """Progress to next level"""
        self.level += 1

        # Empty Groups
        self.playerProjectileGroup.empty()
        self.enemy_group.empty()
        self.enemyProjectileGroup.empty()

        # Generate enemies
        self.level_generator.next_level()
        self.level_generator.generate_level()

    def set_player(self, player: PlayerShip) -> None:
        """Recieves the player ship from the ship select state

        Args:
            player (PlayerShip): the players ship
        """
        self.player = player
        self.player_group.empty()
        self.player_group.add(player)

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

        # Load fonts
        self.assets["font"] = pygame.font.Font(
            "src/assets/fonts/kenvector_future.ttf", 40
        )
        self.assets["med_font"] = pygame.font.Font(
            "src/assets/fonts/kenvector_future.ttf", 30
        )
        self.assets["small_font"] = pygame.font.Font(
            "src/assets/fonts/kenvector_future.ttf", 25
        )

        # Load bomb icon
        self.assets["bomb"] = pygame.image.load(
            f"src/assets/ui/icon-bomb.png"
        ).convert_alpha()

    def update(self, dt: float, events: list[pygame.event.Event]) -> None:
        """Update game objects in game loop

        Args:
            dt (float): time since last frame
        """

        self.background.update(dt)
        self.state_stack.update(dt, events)

    def update_player(self, dt: float) -> None:
        """Public method to allow states to update player ship

        Args:
            dt (float): time since last frame
        """

        self.player_group.update(dt)

    def update_enemies(self, dt: float) -> None:
        """Public method to allow states to update enemies groups

        Args:
            dt (float): time since last frame
        """

        self.enemy_group.update(dt)

    def update_projectiles(self, dt: float) -> None:
        """Public method to allow states to update projectile groups

        Args:
            dt (float): time since last frame
        """

        self.playerProjectileGroup.update(dt, self.enemy_group)
        self.enemyProjectileGroup.update(dt, self.player_group)

    def check_collisions(self) -> None:
        """Public method to determine collisions and calculate damage done"""

        # Check damage to enemies
        for enemy in self.enemy_group:
            for projectile in self.playerProjectileGroup:
                if enemy.shield_active and pygame.sprite.collide_circle(
                    enemy.shield, projectile
                ):
                    enemy.shield.take_damage(projectile.get_shield_damage())
                elif self.is_collision(enemy, projectile):
                    enemy.take_damage(projectile.get_ship_damage())
                    if enemy.is_dead:
                        break
            if enemy.is_dead:
                self.score += int(enemy.get_points() * 10)
                enemy.kill()

        # Check damage to player
        for player in self.player_group:
            for projectile in self.enemyProjectileGroup:
                if player.shield_active and pygame.sprite.collide_circle(
                    player.shield, projectile
                ):
                    player.shield.take_damage(projectile.get_shield_damage())
                elif self.is_collision(player, projectile):
                    player.take_damage(projectile.get_ship_damage())

        # Check ship to ship collision
        for player in self.player_group:
            for enemy in self.enemy_group:
                if self.is_collision(enemy, player):
                    player.take_damage(enemy.max_health)
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
        for player in self.player_group:
            player_projectiles = player.fire()
            if player_projectiles:
                self.playerProjectileGroup.add(player_projectiles)

        # Get enemy projectiles
        for enemy in self.enemy_group:
            enemy_projectiles = enemy.fire()
            if enemy_projectiles:
                self.enemyProjectileGroup.add(enemy_projectiles)

    def remove_offscreen_objects(self) -> None:
        """Deletes objects that have fallen off the screen"""

        for projectile in self.playerProjectileGroup:
            if not self.object_is_onscreen(projectile, [self.is_offscreen_up]):
                projectile.kill()

        for projectile in self.enemyProjectileGroup:
            if not self.object_is_onscreen(projectile, [self.is_offscreen_down]):
                projectile.kill()

        for enemy in self.enemy_group:
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

    def spawn_enemies(self) -> None:
        enemy = self.level_generator.spawn_enemy()
        if enemy:
            self.enemy_group.add(enemy)

    def draw_player(self, window: pygame.Surface) -> None:
        """Public method to allow states to draw player

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        for player in self.player_group:
            player.draw(window)

    def draw_enemies(self, window: pygame.Surface) -> None:
        """Public method to allow states to draw enemies

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        for enemy in self.enemy_group:
            enemy.draw(window)

    def draw_projectiles(self, window: pygame.Surface) -> None:
        """Public method to allow states to draw projectiles

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        self.playerProjectileGroup.draw(window)
        self.enemyProjectileGroup.draw(window)

    def draw_UI(self, window: pygame.Surface) -> None:
        """Draws the UI elements onto the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        (
            health_status,
            shield_status,
            primary_weapon_status,
            secondary_weapon_statuses,
        ) = self.player.get_status()
        x, y, width, height = self.player.get_rect_data()

        # Health status
        if health_status > 0.5:
            color = GREEN
        elif health_status > 0.25:
            color = YELLOW
        else:
            color = RED

        pygame.draw.rect(
            window,
            color,
            pygame.Rect(
                x,
                y + height + 10,
                width * health_status,
                6,
            ),
        )

        # Shield status
        if shield_status:
            shield_surface = pygame.Surface((int(width * shield_status), 10))
            shield_surface.set_alpha(min(150, max(100, shield_status * 255)))
            shield_surface.fill(SHIELD_BLUE)
            window.blit(shield_surface, (x, y + height + 8))

        # Primary weapon status
        if primary_weapon_status:
            pygame.draw.rect(
                window,
                YELLOW,
                pygame.Rect(
                    x - 10,
                    y + (width * (1 - primary_weapon_status)),
                    6,
                    width * primary_weapon_status,
                ),
            )

        # Secondary weapon status
        if secondary_weapon_statuses:
            for index, status in enumerate(secondary_weapon_statuses):
                pygame.draw.rect(
                    window,
                    YELLOW,
                    pygame.Rect(
                        x + width + 4 * (index + 1),
                        y + (width * (1 - status)),
                        2,
                        width * status,
                    ),
                )

        # Draw score
        score_text = self.assets["font"].render(f"Score: {self.score}", 1, GREY)
        window.blit(score_text, (20, 10))

        # Draw lives
        for life in range(self.lives):
            window.blit(self.player.get_icon_sprite(), (16 + life * 30, 60))

        # Draw bombs
        for bomb in range(self.num_bombs):
            window.blit(self.assets["bomb"], (20 + bomb * 15, 95))

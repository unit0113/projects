import pygame

from .ship import Ship
from .player_ship_data import PLAYER_SHIP_DATA
from .settings import (
    WIDTH,
    HEIGHT,
    FRAME_TIME,
    PLAYER_BASE_HP,
    PLAYER_BASE_SHIELD_COOLDOWN,
    PLAYER_BASE_SHIELD_REGEN,
    PLAYER_BASE_SHIELD_STRENGTH,
    PLAYER_BASE_SPEED,
)


class PlayerShip(Ship, pygame.sprite.Sprite):
    def __init__(self, ship_type: str) -> None:
        Ship.__init__(self)
        pygame.sprite.Sprite.__init__(self)

        ship_data = PLAYER_SHIP_DATA[ship_type]

        self.health = ship_data["multipliers"]["hp"] * PLAYER_BASE_HP
        self.max_health = self.health
        self.speed = ship_data["multipliers"]["speed"] * PLAYER_BASE_SPEED
        self.secondary_offsets = ship_data["secondary_offsets"]
        self.projectile_color = ship_data["projectile_color"]

        self.base_shield_strength = (
            ship_data["multipliers"]["shield_strength"] * PLAYER_BASE_SHIELD_STRENGTH
        )
        self.base_shield_cooldown = (
            ship_data["multipliers"]["shield_cooldown"] * PLAYER_BASE_SHIELD_COOLDOWN
        )
        self.base_shield_regen = (
            ship_data["multipliers"]["shield_regen"] * PLAYER_BASE_SHIELD_REGEN
        )

        self.sprites = self.load_sprite_sheet(
            PLAYER_SHIP_DATA[ship_type]["sprite_sheet"], 5, 4
        )
        sprite_number = PLAYER_SHIP_DATA[ship_type]["sprite_sheet"][-2]
        self.sprites["ship_icon"] = pygame.image.load(
            f"src/assets/ui/icon-plane-0{sprite_number}.png"
        ).convert_alpha()

        # Image and animation data
        self.orientation = "level"
        self.frame_index = 0
        self.reset_image()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.last_frame = pygame.time.get_ticks()

        # Weapon data
        self.is_firing_primary = False
        self.is_firing_secondary = False
        self.load_weapons(PLAYER_SHIP_DATA[ship_type], True)

        if PLAYER_SHIP_DATA[ship_type]["start_with_shield"]:
            self.add_shield(2)

    def get_icon_sprite(self) -> pygame.Surface:
        """Returns the ship icon for the UI

        Returns:
            pygame.Surface: icon of ship sprite
        """
        return self.sprites["ship_icon"]

    def update(self, dt: float) -> None:
        """Update game objects in game loop

        Args:
            dt (float): time since last frame
        """

        self.parse_control_input(dt)
        self.animate()
        if self.shield:
            self.shield.update(self.rect.center, self.last_hit)

    def animate(self) -> None:
        """Controls sprite animation of player ship"""

        if pygame.time.get_ticks() > self.last_frame + FRAME_TIME:
            # Reset frame counter and increment frame
            self.last_frame = pygame.time.get_ticks()
            self.frame_index += 1
            if self.frame_index >= len(self.sprites[self.orientation]):
                # Loop animation
                self.frame_index = 0
                # Continue turn
                if self.orientation == "left":
                    self.orientation = "full_left"
                elif self.orientation == "right":
                    self.orientation = "full_right"
            self.reset_image()

    def reset_image(self) -> None:
        """Loads correct image based on animation"""

        self.image = self.sprites[self.orientation][self.frame_index]

    def parse_control_input(self, dt: float) -> None:
        """Moves player based on control input. Sets firing flag if player is firing

        Args:
            dt (float): time since last frame
        """

        self.prev_orientation = self.orientation
        self.orientation = "level"
        self.is_firing_primary = False
        self.is_firing_secondary = False

        keys = pygame.key.get_pressed()
        # Parse Movement
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.player_up(dt)
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.player_down(dt)

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.player_left(dt)
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.player_right(dt)

        # Parse firing state
        if keys[pygame.K_SPACE] or keys[pygame.K_z]:
            self.is_firing_primary = True
            self.is_firing_secondary = True
        elif keys[pygame.K_x]:
            self.is_firing_primary = True
        elif keys[pygame.K_c]:
            self.is_firing_secondary = True

    def player_up(self, dt: float) -> None:
        """Moves player up

        Args:
            dt (float): time since last frame
        """

        self.rect.y = max(0, self.rect.y - self.speed * dt)

    def player_down(self, dt: float) -> None:
        """Moves player down

        Args:
            dt (float): time since last frame
        """

        self.rect.y = min(
            HEIGHT - self.image.get_height(), self.rect.y + self.speed * dt
        )

    def player_left(self, dt: float) -> None:
        """Moves player left

        Args:
            dt (float): time since last frame
        """
        if self.prev_orientation == "full_left":
            self.orientation = "full_left"
        else:
            self.orientation = "left"
        self.rect.x = max(0, self.rect.x - self.speed * dt)

    def player_right(self, dt: float) -> None:
        """Moves player right

        Args:
            dt (float): time since last frame
        """

        if self.prev_orientation == "full_right":
            self.orientation = "full_right"
        else:
            self.orientation = "right"
        self.rect.x = min(WIDTH - self.image.get_width(), self.rect.x + self.speed * dt)

    def fire(self):
        """Returns the projectiles that the ship fired this frame

        Returns:
            Optional[Projectile]: projectiles fired
        """

        projectiles = []
        if self.is_firing_primary:
            for weapon in self.primary_weapons:
                projectile = weapon.fire(self.rect.topleft)
                if projectile:
                    projectiles.append(projectile)

        if self.is_firing_secondary:
            for weapon in self.secondary_weapons:
                side_projectiles = weapon.fire(self.rect.topleft)
                if side_projectiles:
                    for projectile in side_projectiles:
                        if projectile:
                            projectiles.append(projectile)

        return projectiles

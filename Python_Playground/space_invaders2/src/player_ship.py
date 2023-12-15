import pygame

from .ship import Ship
from .player_ship_data import PLAYER_SHIP_DATA
from .settings import WIDTH, HEIGHT, FRAME_TIME
from .shield import Shield


class PlayerShip(Ship):
    def __init__(self, ship_type: str) -> None:
        Ship.__init__(self, PLAYER_SHIP_DATA[ship_type]["hp"])
        self.speed = PLAYER_SHIP_DATA[ship_type]["speed"]

        self.sprites = self.load_sprite_sheet(
            PLAYER_SHIP_DATA[ship_type]["sprite_sheet"], 5, 4
        )

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
        self.load_weapons(PLAYER_SHIP_DATA[ship_type], True)

        self.shield = Shield(100, 1, 1000, self.image.get_width() // 2)

    def update(self, dt: float) -> None:
        """Update game objects in game loop

        Args:
            dt (float): time since last frame
        """

        self.parse_control_input(dt)
        self.animate()
        if self.shield:
            self.shield.update(self.rect.center)

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

    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        window.blit(self.image, self.rect)
        self.shield.draw(window)

    def parse_control_input(self, dt: float) -> None:
        """Moves player based on control input. Sets firing flag if player is firing

        Args:
            dt (float): time since last frame
        """

        self.prev_orientation = self.orientation
        self.orientation = "level"
        self.is_firing_primary = False

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
        if keys[pygame.K_SPACE]:
            self.is_firing_primary = True

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
                projectile = weapon.fire(self.rect.topleft, (0, -1))
                if projectile:
                    projectiles.append(projectile)

        return projectiles

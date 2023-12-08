import pygame

from .ship import Ship
from .player_ship_data import PLAYER_SHIP_DATA
from .settings import WIDTH, HEIGHT, FRAME_TIME


class PlayerShip(Ship):
    def __init__(self, ship_type: str) -> None:
        self.speed = PLAYER_SHIP_DATA[ship_type]["speed"]
        self.max_health = PLAYER_SHIP_DATA[ship_type]["hp"]
        self.health = self.max_health

        self.sprites = self.load_sprite_sheet(
            PLAYER_SHIP_DATA[ship_type]["sprite_sheet"]
        )

        # Image and animation data
        self.orientation = "level"
        self.frame_index = 0
        self.reset_image()
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.last_frame = pygame.time.get_ticks()

        # Weapon data
        self.is_firing_primary = False
        self.load_weapons(PLAYER_SHIP_DATA[ship_type])

    def update(self, dt: float) -> None:
        self.parse_control_input(dt)
        self.animate()

    def animate(self) -> None:
        if pygame.time.get_ticks() > self.last_frame + FRAME_TIME:
            self.last_frame = pygame.time.get_ticks()
            self.frame_index += 1
            if self.frame_index >= len(self.sprites[self.orientation]):
                self.frame_index = 0
            self.reset_image()

    def reset_image(self) -> None:
        self.image = self.sprites[self.orientation][self.frame_index]

    def draw(self, window: pygame.Surface) -> None:
        window.blit(self.image, self.rect)

    def parse_control_input(self, dt: float) -> None:
        self.orientation = "level"
        self.is_firing_primary = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.player_up(dt)
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.player_down(dt)

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.player_left(dt)
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.player_right(dt)

        if keys[pygame.K_SPACE]:
            self.is_firing_primary = True

    def player_up(self, dt: float) -> None:
        self.rect.y = max(0, self.rect.y - self.speed * dt)

    def player_down(self, dt: float) -> None:
        self.rect.y = min(
            HEIGHT - self.image.get_height(), self.rect.y + self.speed * dt
        )

    def player_left(self, dt: float) -> None:
        self.orientation = "left"
        self.rect.x = max(0, self.rect.x - self.speed * dt)

    def player_right(self, dt: float) -> None:
        self.orientation = "right"
        self.rect.x = min(WIDTH - self.image.get_width(), self.rect.x + self.speed * dt)

    def fire(self):
        projectiles = []
        if self.is_firing_primary:
            for weapon in self.primary_weapons:
                projectile = weapon.fire(self.rect.topleft, (0, -1))
                if projectile:
                    projectiles.append(projectile)

        return projectiles

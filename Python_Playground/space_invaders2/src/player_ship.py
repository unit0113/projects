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
        self.primary_weapons = PLAYER_SHIP_DATA[ship_type]["primary_weapons"]
        self.parse_weapon_loadout()

    def parse_weapon_loadout(self) -> None:
        for location, weapon in self.primary_weapons:
            pass

    def update(self, dt: float) -> None:
        self.move(dt)
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

    def move(self, dt: float):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.player_up(dt)
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.player_down(dt)

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.player_left(dt)
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.player_right(dt)

    def player_up(self, dt: float):
        self.rect.y = max(0, self.rect.y - self.speed * dt)

    def player_down(self, dt: float):
        self.rect.y = min(
            HEIGHT - self.image.get_height(), self.rect.y + self.speed * dt
        )

    def player_left(self, dt: float):
        self.rect.x = max(0, self.rect.x - self.speed * dt)

    def player_right(self, dt: float):
        self.rect.x = min(WIDTH - self.image.get_width(), self.rect.x + self.speed * dt)

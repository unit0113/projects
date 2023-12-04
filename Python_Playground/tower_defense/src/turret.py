import pygame
import math

from src.settings import TILESIZE, ANIMATION_STEPS, ANIMATION_DELAY
from src.turret_data import TURRET_DATA


class Turret(pygame.sprite.Sprite):
    def __init__(
        self, sprite_sheets: list[pygame.surface.Surface], pos: tuple[int, int]
    ) -> None:
        pygame.sprite.Sprite.__init__(self)
        self.upgrade_level = 1
        self.range = TURRET_DATA[self.upgrade_level - 1].get("range")
        self.cooldown = TURRET_DATA[self.upgrade_level - 1].get("cooldown")
        self.damage = TURRET_DATA[self.upgrade_level - 1].get("damage")

        self.tile_x = pos[0] // TILESIZE
        self.tile_y = pos[1] // TILESIZE
        # Calculate center coord
        self.pos = ((self.tile_x + 0.5) * TILESIZE, (self.tile_y + 0.5) * TILESIZE)

        self.animation_lists = self.extract_sprite_sheets(sprite_sheets)
        self.frame_index = 0
        self.timer = pygame.time.get_ticks()
        self.original_image = self.animation_lists[self.upgrade_level - 1][
            self.frame_index
        ]
        self.angle = 90
        self.image = pygame.transform.rotate(self.original_image, self.angle)

        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        self.update_range_circle()

        self.last_shot = pygame.time.get_ticks()
        self.selected = False
        self.target = None

    def extract_sprite_sheets(
        self, sprite_sheets: list[pygame.surface.Surface]
    ) -> list[list[pygame.surface.Surface]]:
        animation_lists = []
        for sprite_sheet in sprite_sheets:
            size = sprite_sheet.get_height()
            animation_list = []

            for x in range(ANIMATION_STEPS):
                animation_list.append(sprite_sheet.subsurface(x * size, 0, size, size))
            animation_lists.append(animation_list)

        return animation_lists

    def play_animation(self) -> None:
        self.original_image = self.animation_lists[self.upgrade_level - 1][
            self.frame_index
        ]
        if pygame.time.get_ticks() - self.timer > ANIMATION_DELAY:
            self.timer = pygame.time.get_ticks()
            self.frame_index += 1
            if self.frame_index >= len(self.animation_lists[self.upgrade_level - 1]):
                self.frame_index = 0
                self.last_shot = pygame.time.get_ticks()
                self.target = None

    def select_target(self, enemy_group: pygame.sprite.Group) -> int:
        reward = 0
        for enemy in enemy_group:
            x_dist = enemy.pos[0] - self.pos[0]
            y_dist = enemy.pos[1] - self.pos[1]
            dist = math.sqrt(x_dist**2 + y_dist**2)
            if dist < self.range:
                self.target = enemy
                self.angle = math.degrees(math.atan2(-y_dist, x_dist))
                reward = self.target.take_damage(self.damage)
        return reward

    def update(self, enemy_group: pygame.sprite.Group) -> int:
        if self.target:
            self.play_animation()
            return 0

        elif pygame.time.get_ticks() - self.last_shot >= self.cooldown:
            return self.select_target(enemy_group)
        return 0

    def draw(self, window: pygame.surface.Surface) -> None:
        if self.selected:
            window.blit(self.range_image, self.range_rect)
        self.image = pygame.transform.rotate(self.original_image, self.angle - 90)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        window.blit(self.image, self.rect)

    def upgrade(self) -> None:
        if not self.can_upgrade():
            return

        self.upgrade_level += 1
        self.original_image = self.animation_lists[self.upgrade_level - 1][
            self.frame_index
        ]

        self.range = TURRET_DATA[self.upgrade_level - 1].get("range")
        self.cooldown = TURRET_DATA[self.upgrade_level - 1].get("cooldown")
        self.damage = TURRET_DATA[self.upgrade_level - 1].get("damage")

        self.update_range_circle()

    def can_upgrade(self) -> bool:
        return self.upgrade_level < len(TURRET_DATA)

    def update_range_circle(self):
        self.range_image = pygame.Surface((self.range * 2, self.range * 2))
        self.range_image.fill((0, 0, 0))
        self.range_image.set_colorkey((0, 0, 0))
        pygame.draw.circle(
            self.range_image, "grey100", (self.range, self.range), self.range
        )
        self.range_image.set_alpha(100)
        self.range_rect = self.range_image.get_rect()
        self.range_rect.center = self.rect.center

import pygame

from src.settings import (
    WIDTH,
    HEIGHT,
    TILESIZE,
    NUM_COLS,
    HEALTH,
    MONEY,
    BUY_COST,
    UPGRADE_COST,
)
from src.enemy_factory import EnemyFactory
from src.ui import UI
from src.settings import LEVEL_COMPLETE_REWARD


class World:
    def __init__(
        self,
        map_image: pygame.surface.Surface,
        data: dict,
        enemy_images: dict[str, pygame.surface.Surface],
    ) -> None:
        self.level = 0
        self.health = HEALTH
        self.money = MONEY

        self.ui = UI()
        self.image = map_image
        self.data = data
        self._process_data()
        self.enemy_factory = EnemyFactory(enemy_images, self.waypoints)

    def _process_data(self) -> None:
        for layer in self.data["layers"]:
            if layer["name"] == "tilemap":
                self.tilemap = [True if tile == 7 else False for tile in layer["data"]]
            elif layer["name"] == "waypoints":
                for obj in layer["objects"]:
                    raw_waypoints = obj["polyline"]
                    self._process_raw_waypoints(raw_waypoints)

    def _process_raw_waypoints(self, raw_waypoints: list[dict]) -> None:
        self.waypoints = []
        for point in raw_waypoints:
            self.waypoints.append([point.get("x"), point.get("y")])

    def draw(self, window) -> None:
        window.blit(self.image, (0, 0))

    def valid_turret_location(self, mouse_pos: tuple[int, int]) -> bool:
        if not 0 < mouse_pos[0] < WIDTH or not 0 < mouse_pos[1] < HEIGHT:
            return False

        mouse_tile_x = mouse_pos[0] // TILESIZE
        mouse_tile_y = mouse_pos[1] // TILESIZE
        tilemap_index = NUM_COLS * mouse_tile_x + mouse_tile_y
        valid = self.tilemap[tilemap_index]
        self.tilemap[tilemap_index] = False
        return valid

    def reset_turret_location(self, mouse_pos: tuple[int, int]) -> None:
        mouse_tile_x = mouse_pos[0] // TILESIZE
        mouse_tile_y = mouse_pos[1] // TILESIZE
        tilemap_index = NUM_COLS * mouse_tile_x + mouse_tile_y
        self.tilemap[tilemap_index] = True

    def get_enemies(self) -> pygame.sprite.Group:
        return self.enemy_factory.get_enemies(self.level)

    def draw_ui(self, window) -> None:
        self.ui.draw_text(
            window, f"Health: {self.health}", self.ui.text_font, "grey100", 0, 0
        )
        self.ui.draw_text(
            window, f"Money: {self.money}", self.ui.text_font, "grey100", 0, 30
        )
        self.ui.draw_text(
            window, f"Level {self.level}", self.ui.text_font, "grey100", 0, 60
        )

    def purchase_new_turret(self) -> bool:
        if self.money >= BUY_COST:
            self.money -= BUY_COST
            return True
        return False

    def purchase_turret_upgrade(self) -> bool:
        if self.money >= UPGRADE_COST:
            self.money -= UPGRADE_COST
            return True
        return False

    def updateMoney(self, money: int) -> None:
        self.money += money

    def take_damage(self, dmg: int) -> None:
        self.health -= dmg

    def new_level(self) -> None:
        if self.level > 0:
            self.money += LEVEL_COMPLETE_REWARD
        self.level += 1

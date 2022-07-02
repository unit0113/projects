import pygame


STORE_HEALTH_BASE_COST = 10000
STORE_SPEED_BASE_COST = 10000
STORE_DAMAGE_BASE_COST = 10000
STORE_LASER_REGEN_BASE_COST = 10000
STORE_LASER_MAX_CHARGE_BASE_COST = 10000
STORE_LASER_COST_BASE_COST = 10000
STORE_INFLATION = 1.1
STORE_LIVES_COST = 25000
STORE_LASER_UPGRADE_COST = 50000
UPGRADE_PERCENT = 0.1
STORE_WINDOW_PADDING = 50


class Store:
    def __init__(self, window, player_ship):
        self.window = window
        self.player_ship = player_ship
        self.store_title_font = pygame.font.SysFont('verdana', 40, bold=True)
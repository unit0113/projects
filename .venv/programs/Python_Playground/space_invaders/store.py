import pygame
from space_invaders import HEIGHT, WIDTH, FPS, WHITE, STEEL
from ship import SHIELD_POST_HIT_COOLDOWN

STORE_HEALTH_BASE_COST = 10000
STORE_SPEED_BASE_COST = 10000
STORE_DAMAGE_BASE_COST = 10000
STORE_LASER_REGEN_BASE_COST = 10000
STORE_LASER_MAX_CHARGE_BASE_COST = 10000
STORE_LASER_COST_BASE_COST = 10000
STORE_SHIELD_REGEN_BASE_COST = 10000
STORE_SHIELD_COOLDOWN_BASE_COST = 10000

STORE_INFLATION = 0.25
STORE_LIVES_COST = 25000
STORE_LASER_UPGRADE_COST = 50000
STORE_SHIELD_LEVEL_UPGRADE_COST = 25000
UPGRADE_PERCENT = 0.1
STORE_WINDOW_PADDING = 50
SPACER = 50


class Store:
    def __init__(self, window, background, player_ship):
        self.window = window
        self.background = background
        self.player_ship = player_ship
        self.store_title_font = pygame.font.SysFont('verdana', 40, bold=True)
        self.store_font = pygame.font.SysFont('verdana', 25, bold=True)

    def open_store(self, credits):
        credits_spent = 0

        while True:
            self._draw_store(credits-credits_spent)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    return credits_spent

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    pygame.quit()
                    quit()


    def _draw_store(self, credits):
        self.window.blit(self.background, (0, 0))
        store_rect = pygame.Rect(STORE_WINDOW_PADDING, STORE_WINDOW_PADDING, WIDTH - 2 * STORE_WINDOW_PADDING, HEIGHT - 2 * STORE_WINDOW_PADDING)
        pygame.draw.rect(self.window, STEEL, store_rect, border_radius=10)

        store_title_text = self.store_title_font.render('Upgrade Store', 1, WHITE)
        self.window.blit(store_title_text, (WIDTH // 2 - store_title_text.get_width() // 2, STORE_WINDOW_PADDING + store_title_text.get_height() // 2))

        col1_start, col2_start = STORE_WINDOW_PADDING * 3, WIDTH // 2 + STORE_WINDOW_PADDING
        col1_num_start, col2_num_start = WIDTH // 2 - STORE_WINDOW_PADDING, WIDTH - STORE_WINDOW_PADDING * 3

        spacer = 3
        store_credits_text = self.store_title_font.render(f'Avaliable Credits: {credits}', 1, WHITE)
        self.window.blit(store_credits_text, (WIDTH // 2 - store_credits_text.get_width() // 2, STORE_WINDOW_PADDING + store_credits_text.get_height() // 2 + SPACER * spacer))        

        # Row 1
        spacer += 2
        # Left
        store_health_text = self.store_font.render(f"Hitpoints", 1, WHITE)
        self.window.blit(store_health_text, (col1_start, STORE_WINDOW_PADDING + store_health_text.get_height() // 2 + SPACER * spacer))
        store_health_text_2 = self.store_font.render(f"{self.player_ship.max_health:.0f}", 1, WHITE)
        self.window.blit(store_health_text_2, (col1_num_start - store_health_text_2.get_width(), STORE_WINDOW_PADDING + store_health_text_2.get_height() // 2 + SPACER * spacer))

        # Right
        store_speed_text = self.store_font.render(f"Speed", 1, WHITE)
        self.window.blit(store_speed_text, (col2_start, STORE_WINDOW_PADDING + store_speed_text.get_height() // 2 + SPACER * spacer))
        store_speed_text_2 = self.store_font.render(f"{self.player_ship.speed:.0f}", 1, WHITE)
        self.window.blit(store_speed_text_2, (col2_num_start - store_speed_text_2.get_width(), STORE_WINDOW_PADDING + store_speed_text_2.get_height() // 2 + SPACER * spacer))

        # Row 2
        spacer += 1
        # Left
        store_base_dmg_text = self.store_font.render(f"Base Damage", 1, WHITE)
        self.window.blit(store_base_dmg_text, (col1_start, STORE_WINDOW_PADDING + store_base_dmg_text.get_height() // 2 + SPACER * spacer))
        store_store_base_dmg_text_2 = self.store_font.render(f"{self.player_ship.base_damage:.0f}", 1, WHITE)
        self.window.blit(store_store_base_dmg_text_2, (col1_num_start - store_store_base_dmg_text_2.get_width(), STORE_WINDOW_PADDING + store_store_base_dmg_text_2.get_height() // 2 + SPACER * spacer))

        # Right
        if self.player_ship.shield_level:
            store_shield_cooldown_text = self.store_font.render(f"Shield Cooldown", 1, WHITE)
            self.window.blit(store_shield_cooldown_text, (col2_start, STORE_WINDOW_PADDING + store_shield_cooldown_text.get_height() // 2 + SPACER * spacer))
            store_shield_cooldown_text_2 = self.store_font.render(f"{SHIELD_POST_HIT_COOLDOWN / self.player_ship.shield_cooldown_modifier:.0f}", 1, WHITE)
            self.window.blit(store_shield_cooldown_text_2, (col2_num_start - store_shield_cooldown_text_2.get_width(), STORE_WINDOW_PADDING + store_shield_cooldown_text_2.get_height() // 2 + SPACER * spacer))

        # Row 3
        spacer += 1
        # Left
        store_laser_regen_text = self.store_font.render(f"Laser Regen", 1, WHITE)
        self.window.blit(store_laser_regen_text, (col1_start, STORE_WINDOW_PADDING + store_laser_regen_text.get_height() // 2 + SPACER * spacer))
        store_laser_regen_text_2 = self.store_font.render(f"{self.player_ship.laser_regen:.0f}", 1, WHITE)
        self.window.blit(store_laser_regen_text_2, (col1_num_start - store_laser_regen_text_2.get_width(), STORE_WINDOW_PADDING + store_laser_regen_text_2.get_height() // 2 + SPACER * spacer))

        # Right
        if self.player_ship.shield_level:
            store_shield_regen_text = self.store_font.render(f"Shield Regen", 1, WHITE)
            self.window.blit(store_shield_regen_text, (col2_start, STORE_WINDOW_PADDING + store_shield_regen_text.get_height() // 2 + SPACER * spacer))
            store_shield_regen_text_2 = self.store_font.render(f"{self.player_ship.shield_regen:.0f}", 1, WHITE)
            self.window.blit(store_shield_regen_text_2, (col2_num_start - store_shield_regen_text_2.get_width(), STORE_WINDOW_PADDING + store_shield_regen_text_2.get_height() // 2 + SPACER * spacer))

        # Row 4
        spacer += 1
        # Left
        store_laser_max_chrg_text = self.store_font.render(f"Laser Max Charge", 1, WHITE)
        self.window.blit(store_laser_max_chrg_text, (col1_start, STORE_WINDOW_PADDING + store_laser_max_chrg_text.get_height() // 2 + SPACER * spacer))
        store_laser_max_chrg_text_2 = self.store_font.render(f"{self.player_ship.laser_max_charge:.0f}", 1, WHITE)
        self.window.blit(store_laser_max_chrg_text_2, (col1_num_start - store_laser_max_chrg_text_2.get_width(), STORE_WINDOW_PADDING + store_laser_max_chrg_text_2.get_height() // 2 + SPACER * spacer))

        # Right
        store_shield_level_text = self.store_font.render(f"Shield Level", 1, WHITE)
        self.window.blit(store_shield_level_text, (col2_start, STORE_WINDOW_PADDING + store_shield_level_text.get_height() // 2 + SPACER * spacer))
        store_shield_level_text_2 = self.store_font.render(f"{self.player_ship.shield_level:.0f}", 1, WHITE)
        self.window.blit(store_shield_level_text_2, (col2_num_start - store_shield_level_text_2.get_width(), STORE_WINDOW_PADDING + store_shield_level_text_2.get_height() // 2 + SPACER * spacer))

        # Row 5
        spacer += 1
        # Left
        store_laser_cost_text = self.store_font.render(f"Laser Charge Usage", 1, WHITE)
        self.window.blit(store_laser_cost_text, (col1_start, STORE_WINDOW_PADDING + store_laser_cost_text.get_height() // 2 + SPACER * spacer))
        store_laser_cost_text_2 = self.store_font.render(f"{self.player_ship.laser_cost:.0f}", 1, WHITE)
        self.window.blit(store_laser_cost_text_2, (col1_num_start - store_laser_cost_text_2.get_width(), STORE_WINDOW_PADDING + store_laser_cost_text_2.get_height() // 2 + SPACER * spacer))

        # Right
        store_laser_level_text = self.store_font.render(f"Laser Level", 1, WHITE)
        self.window.blit(store_laser_level_text, (col2_start, STORE_WINDOW_PADDING + store_laser_level_text.get_height() // 2 + SPACER * spacer))
        store_laser_level_text_2 = self.store_font.render(f"{self.player_ship.laser_type_current_index + 1:.0f}", 1, WHITE)
        self.window.blit(store_laser_level_text_2, (col2_num_start - store_laser_level_text_2.get_width(), STORE_WINDOW_PADDING + store_laser_level_text_2.get_height() // 2 + SPACER * spacer))

        spacer += 4
        instructions_text = self.store_title_font.render('Press C to continue.', 1, WHITE)
        self.window.blit(instructions_text, (WIDTH // 2 - instructions_text.get_width() // 2, STORE_WINDOW_PADDING + instructions_text.get_height() // 2 + SPACER * spacer))


        pygame.display.update()
import pygame
from settings import HEIGHT, WIDTH, FPS, WHITE, STEEL, RED, SHIELD_POST_HIT_COOLDOWN

STORE_HEALTH_BASE_COST = 10_000
STORE_SPEED_BASE_COST = 10_000
STORE_DAMAGE_BASE_COST = 10_000
STORE_LASER_REGEN_BASE_COST = 10_000
STORE_LASER_MAX_CHARGE_BASE_COST = 10_000
STORE_LASER_COST_BASE_COST = 10_000
STORE_SHIELD_REGEN_BASE_COST = 10_000
STORE_SHIELD_COOLDOWN_BASE_COST = 10_000
STORE_SMISSILE_DAMAGE_BASE_COST = 10_000

STORE_INFLATION = 0.25
STORE_LIVES_COST = 25_000
STORE_LIVES_PER_LEVEL_ADDITIONAL_COST = 5_000
STORE_LASER_LEVEL_UPGRADE_COST = 50_000
STORE_SHIELD_LEVEL_UPGRADE_COST = 25_000
STORE_WINDOW_PADDING = 50
SPACER = 50


class Store:
    def __init__(self, window, background, player_ship):
        self.window = window
        self.background = background
        self.player_ship = player_ship
        self.credits = 0
        self.store_title_font = pygame.font.SysFont('verdana', 40, bold=True)
        self.store_font = pygame.font.SysFont('verdana', 25, bold=True)
        self.increase_image = pygame.image.load('Python_Playground\space_invaders\Assets\increase.png').convert_alpha()
        self.increase_image = pygame.transform.scale(self.increase_image, (30,30))
        self.display_warning = False
        self.warning_text = self.store_font.render('', 1, RED)
        self.create_rects()

    def create_rects(self):
        self.rects = []
        spacer = 5
        self.health_rect = self.increase_image.get_rect(topleft=(STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))
        self.rects.append(self.health_rect)
        self.speed_rect = self.increase_image.get_rect(topleft=(WIDTH - STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))
        self.rects.append(self.speed_rect)

        spacer += 1
        self.base_dmg_rect = self.increase_image.get_rect(topleft=(STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))
        self.rects.append(self.base_dmg_rect)
        self.shield_cooldown_rect = self.increase_image.get_rect(topleft=(WIDTH - STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))
        self.rects.append(self.shield_cooldown_rect)
        
        spacer += 1
        self.laser_regen_rect = self.increase_image.get_rect(topleft=(STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))
        self.rects.append(self.laser_regen_rect)
        self.shield_regen_rect = self.increase_image.get_rect(topleft=(WIDTH - STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))
        self.rects.append(self.shield_regen_rect)
                
        spacer += 1
        self.laser_max_charge_rect = self.increase_image.get_rect(topleft=(STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))
        self.rects.append(self.laser_max_charge_rect)
        self.shield_level_rect = self.increase_image.get_rect(topleft=(WIDTH - STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))
        self.rects.append(self.shield_level_rect)
                        
        spacer += 1
        self.laser_cost_rect = self.increase_image.get_rect(topleft=(STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))
        self.rects.append(self.laser_cost_rect)
        self.laser_level_rect = self.increase_image.get_rect(topleft=(WIDTH - STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))
        self.rects.append(self.laser_level_rect)

        spacer += 1
        self.lives_rect = self.increase_image.get_rect(topleft=(STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))
        self.rects.append(self.lives_rect)

    def open_store(self, new_credits, lives, level):
        self.credits += new_credits
        self.lives = lives
        clock = pygame.time.Clock()
        self.display_warning = False

        while True:
            clock.tick(FPS)
            self._draw_store()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    return self.lives

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    pygame.quit()
                    quit()

                if event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    for rect in self.rects:
                        if rect.collidepoint(pos):
                            match rect:
                                case self.health_rect:
                                    self._increase_health()
                                case self.speed_rect:
                                    self._increase_speed()
                                case self.base_dmg_rect:
                                    self._increase_base_dmg()
                                case self.shield_cooldown_rect:
                                    if self.player_ship.shield_level:
                                        self._increase_shield_cooldown()
                                case self.laser_regen_rect:
                                    self._increase_laser_regen()
                                case self.shield_regen_rect:
                                    if self.player_ship.shield_level:
                                        self._increase_shield_regen()
                                case self.laser_max_charge_rect:
                                    self._increase_laser_max_charge()
                                case self.shield_level_rect:
                                    self._increase_shield_level()
                                case self.laser_cost_rect:
                                    self._increase_laser_cost()
                                case self.laser_level_rect:
                                    self._increase_laser_level()
                                case self.lives_rect:
                                    self._increase_lives(level)

    def _draw_store(self):
        self.window.blit(self.background, (0, 0))
        store_rect = pygame.Rect(STORE_WINDOW_PADDING, STORE_WINDOW_PADDING, WIDTH - 2 * STORE_WINDOW_PADDING, HEIGHT - 2 * STORE_WINDOW_PADDING)
        pygame.draw.rect(self.window, STEEL, store_rect, border_radius=10)

        store_title_text = self.store_title_font.render('Upgrade Store', 1, WHITE)
        self.window.blit(store_title_text, (WIDTH // 2 - store_title_text.get_width() // 2, STORE_WINDOW_PADDING + store_title_text.get_height() // 2))

        col1_start, col2_start = STORE_WINDOW_PADDING * 3, WIDTH // 2 + STORE_WINDOW_PADDING
        col1_num_start, col2_num_start = WIDTH // 2 - STORE_WINDOW_PADDING, WIDTH - STORE_WINDOW_PADDING * 3

        spacer = 3
        store_credits_text = self.store_title_font.render(f'Avaliable Credits: {self.credits}', 1, WHITE)
        self.window.blit(store_credits_text, (WIDTH // 2 - store_credits_text.get_width() // 2, STORE_WINDOW_PADDING + store_credits_text.get_height() // 2 + SPACER * spacer))        

        # Row 1
        spacer += 2
        # Left
        store_health_text = self.store_font.render(f"Hitpoints", 1, WHITE)
        self.window.blit(store_health_text, (col1_start, STORE_WINDOW_PADDING + store_health_text.get_height() // 2 + SPACER * spacer))
        store_health_text_2 = self.store_font.render(f"{self.player_ship.max_health:.0f}", 1, WHITE)
        self.window.blit(store_health_text_2, (col1_num_start - store_health_text_2.get_width(), STORE_WINDOW_PADDING + store_health_text_2.get_height() // 2 + SPACER * spacer))
        self.window.blit(self.increase_image, (self.health_rect.x, self.health_rect.y))

        # Right
        store_speed_text = self.store_font.render(f"Speed", 1, WHITE)
        self.window.blit(store_speed_text, (col2_start, STORE_WINDOW_PADDING + store_speed_text.get_height() // 2 + SPACER * spacer))
        store_speed_text_2 = self.store_font.render(f"{self.player_ship.speed:.0f}", 1, WHITE)
        self.window.blit(store_speed_text_2, (col2_num_start - store_speed_text_2.get_width(), STORE_WINDOW_PADDING + store_speed_text_2.get_height() // 2 + SPACER * spacer))
        self.window.blit(self.increase_image, (self.speed_rect.x, self.speed_rect.y))

        # Row 2
        spacer += 1
        # Left
        store_base_dmg_text = self.store_font.render(f"Base Damage", 1, WHITE)
        self.window.blit(store_base_dmg_text, (col1_start, STORE_WINDOW_PADDING + store_base_dmg_text.get_height() // 2 + SPACER * spacer))
        store_store_base_dmg_text_2 = self.store_font.render(f"{self.player_ship.base_damage:.0f}", 1, WHITE)
        self.window.blit(store_store_base_dmg_text_2, (col1_num_start - store_store_base_dmg_text_2.get_width(), STORE_WINDOW_PADDING + store_store_base_dmg_text_2.get_height() // 2 + SPACER * spacer))
        self.window.blit(self.increase_image, (self.base_dmg_rect.x, self.base_dmg_rect.y))

        # Right
        if self.player_ship.shield_level:
            store_shield_cooldown_text = self.store_font.render(f"Shield Cooldown", 1, WHITE)
            self.window.blit(store_shield_cooldown_text, (col2_start, STORE_WINDOW_PADDING + store_shield_cooldown_text.get_height() // 2 + SPACER * spacer))
            store_shield_cooldown_text_2 = self.store_font.render(f"{SHIELD_POST_HIT_COOLDOWN / self.player_ship.shield_cooldown_modifier:.0f}", 1, WHITE)
            self.window.blit(store_shield_cooldown_text_2, (col2_num_start - store_shield_cooldown_text_2.get_width(), STORE_WINDOW_PADDING + store_shield_cooldown_text_2.get_height() // 2 + SPACER * spacer))
            self.window.blit(self.increase_image, (WIDTH - STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))

        # Row 3
        spacer += 1
        # Left
        store_laser_regen_text = self.store_font.render(f"Laser Regen", 1, WHITE)
        self.window.blit(store_laser_regen_text, (col1_start, STORE_WINDOW_PADDING + store_laser_regen_text.get_height() // 2 + SPACER * spacer))
        store_laser_regen_text_2 = self.store_font.render(f"{self.player_ship.laser_regen:.0f}", 1, WHITE)
        self.window.blit(store_laser_regen_text_2, (col1_num_start - store_laser_regen_text_2.get_width(), STORE_WINDOW_PADDING + store_laser_regen_text_2.get_height() // 2 + SPACER * spacer))
        self.window.blit(self.increase_image, (STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))

        # Right
        if self.player_ship.shield_level:
            store_shield_regen_text = self.store_font.render(f"Shield Regen", 1, WHITE)
            self.window.blit(store_shield_regen_text, (col2_start, STORE_WINDOW_PADDING + store_shield_regen_text.get_height() // 2 + SPACER * spacer))
            store_shield_regen_text_2 = self.store_font.render(f"{FPS * self.player_ship.shield_regen:.0f}", 1, WHITE)
            self.window.blit(store_shield_regen_text_2, (col2_num_start - store_shield_regen_text_2.get_width(), STORE_WINDOW_PADDING + store_shield_regen_text_2.get_height() // 2 + SPACER * spacer))
            self.window.blit(self.increase_image, (WIDTH - STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))

        # Row 4
        spacer += 1
        # Left
        store_laser_max_chrg_text = self.store_font.render(f"Laser Max Charge", 1, WHITE)
        self.window.blit(store_laser_max_chrg_text, (col1_start, STORE_WINDOW_PADDING + store_laser_max_chrg_text.get_height() // 2 + SPACER * spacer))
        store_laser_max_chrg_text_2 = self.store_font.render(f"{self.player_ship.laser_max_charge:.0f}", 1, WHITE)
        self.window.blit(store_laser_max_chrg_text_2, (col1_num_start - store_laser_max_chrg_text_2.get_width(), STORE_WINDOW_PADDING + store_laser_max_chrg_text_2.get_height() // 2 + SPACER * spacer))
        self.window.blit(self.increase_image, (STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))

        # Right
        store_shield_level_text = self.store_font.render(f"Shield Level", 1, WHITE)
        self.window.blit(store_shield_level_text, (col2_start, STORE_WINDOW_PADDING + store_shield_level_text.get_height() // 2 + SPACER * spacer))
        store_shield_level_text_2 = self.store_font.render(f"{self.player_ship.shield_level:.0f}", 1, WHITE)
        self.window.blit(store_shield_level_text_2, (col2_num_start - store_shield_level_text_2.get_width(), STORE_WINDOW_PADDING + store_shield_level_text_2.get_height() // 2 + SPACER * spacer))
        self.window.blit(self.increase_image, (WIDTH - STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))

        # Row 5
        spacer += 1
        # Left
        store_laser_cost_text = self.store_font.render(f"Laser Charge Usage", 1, WHITE)
        self.window.blit(store_laser_cost_text, (col1_start, STORE_WINDOW_PADDING + store_laser_cost_text.get_height() // 2 + SPACER * spacer))
        store_laser_cost_text_2 = self.store_font.render(f"{self.player_ship.laser_cost:.0f}", 1, WHITE)
        self.window.blit(store_laser_cost_text_2, (col1_num_start - store_laser_cost_text_2.get_width(), STORE_WINDOW_PADDING + store_laser_cost_text_2.get_height() // 2 + SPACER * spacer))
        self.window.blit(self.increase_image, (STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))

        # Right
        store_laser_level_text = self.store_font.render(f"Laser Level", 1, WHITE)
        self.window.blit(store_laser_level_text, (col2_start, STORE_WINDOW_PADDING + store_laser_level_text.get_height() // 2 + SPACER * spacer))
        store_laser_level_text_2 = self.store_font.render(f"{self.player_ship.laser_level + 1:.0f}", 1, WHITE)
        self.window.blit(store_laser_level_text_2, (col2_num_start - store_laser_level_text_2.get_width(), STORE_WINDOW_PADDING + store_laser_level_text_2.get_height() // 2 + SPACER * spacer))
        self.window.blit(self.increase_image, (WIDTH - STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))

        spacer += 1
        # Left
        store_lives_text = self.store_font.render(f"Lives Remaining", 1, WHITE)
        self.window.blit(store_lives_text, (col1_start, STORE_WINDOW_PADDING + store_lives_text.get_height() // 2 + SPACER * spacer))
        store_lives_text_2 = self.store_font.render(f"{self.lives:.0f}", 1, WHITE)
        self.window.blit(store_lives_text_2, (col1_num_start - store_lives_text_2.get_width(), STORE_WINDOW_PADDING + store_lives_text_2.get_height() // 2 + SPACER * spacer))
        self.window.blit(self.increase_image, (STORE_WINDOW_PADDING * 2, STORE_WINDOW_PADDING + self.increase_image.get_height() // 2 + SPACER * spacer))

        spacer += 3
        instructions_text = self.store_title_font.render('Press C to continue.', 1, WHITE)
        self.window.blit(instructions_text, (WIDTH // 2 - instructions_text.get_width() // 2, STORE_WINDOW_PADDING + instructions_text.get_height() // 2 + SPACER * spacer))

        spacer += 2
        if self.display_warning:
            self.window.blit(self.warning_text, (WIDTH // 2 - self.warning_text.get_width() // 2, STORE_WINDOW_PADDING + self.warning_text.get_height() // 2 + SPACER * spacer))

        pygame.display.update()

    def _get_credit_multiplier(self, num_upgrades):
        return (1 + STORE_INFLATION) ** num_upgrades

    def _generate_warning(self, req_credits):
        self.warning_text = self.store_font.render(f'Unable to purchase upgrade. {self.credits} credits available, {req_credits} credits required.', 1, RED)
        self.display_warning = True

    def _increase_health(self):
        req_credits = int(STORE_HEALTH_BASE_COST * self._get_credit_multiplier(self.player_ship.health_upgrades))
        if self.credits > req_credits:
            self.display_warning = False
            self.player_ship.health_upgrades += 1
            self.credits -= req_credits

        else:
            self._generate_warning(req_credits)

    def _increase_speed(self):
        req_credits = int(STORE_SPEED_BASE_COST * self._get_credit_multiplier(self.player_ship.speed_upgrades))
        if self.credits > req_credits:
            self.display_warning = False
            self.player_ship.speed_upgrades += 1
            self.credits -= req_credits

        else:
            self._generate_warning(req_credits)

    def _increase_base_dmg(self):
        req_credits = int(STORE_DAMAGE_BASE_COST * self._get_credit_multiplier(self.player_ship.damage_upgrades))
        if self.credits > req_credits:
            self.display_warning = False
            self.player_ship.damage_upgrades += 1
            self.credits -= req_credits

        else:
            self._generate_warning(req_credits)

    def _increase_shield_cooldown(self):
        req_credits = int(STORE_SHIELD_COOLDOWN_BASE_COST * self._get_credit_multiplier(self.player_ship.shield_cooldown_upgrades))
        if self.credits > req_credits:
            self.display_warning = False
            self.player_ship.shield_cooldown_upgrades += 1
            self.credits -= req_credits

        else:
            self._generate_warning(req_credits)

    def _increase_laser_regen(self):
        req_credits = int(STORE_LASER_REGEN_BASE_COST * self._get_credit_multiplier(self.player_ship.laser_regen_upgrades))
        if self.credits > req_credits:
            self.display_warning = False
            self.player_ship.laser_regen_upgrades += 1
            self.credits -= req_credits

        else:
            self._generate_warning(req_credits)

    def _increase_shield_regen(self):
        req_credits = int(STORE_SHIELD_REGEN_BASE_COST * self._get_credit_multiplier(self.player_ship.shield_regen_upgrades))
        if self.credits > req_credits:
            self.display_warning = False
            self.player_ship.shield_regen_upgrades += 1
            self.credits -= req_credits

        else:
            self._generate_warning(req_credits)

    def _increase_laser_max_charge(self):
        req_credits = int(STORE_LASER_MAX_CHARGE_BASE_COST * self._get_credit_multiplier(self.player_ship.laser_max_charge_upgrades))
        if self.credits > req_credits:
            self.display_warning = False
            self.player_ship.laser_max_charge_upgrades += 1
            self.credits -= req_credits

        else:
            self._generate_warning(req_credits)

    def _increase_laser_cost(self):
        req_credits = int(STORE_LASER_COST_BASE_COST * self._get_credit_multiplier(self.player_ship.laser_cost_upgrades))
        if self.credits > req_credits:
            self.display_warning = False
            self.player_ship.laser_cost_upgrades += 1
            self.credits -= req_credits

        else:
            self._generate_warning(req_credits)

    def _increase_shield_level(self):
        if self.player_ship.at_max_shield_level:
            self.warning_text = self.store_font.render(f'Shield level at maximum.', 1, RED)
            self.display_warning = True
            return

        req_credits = int(STORE_SHIELD_LEVEL_UPGRADE_COST * (1 + self.player_ship.shield_level))
        if self.credits > req_credits:
            self.display_warning = False
            self.player_ship.shield_level += 1
            self.credits -= req_credits

        else:
            self._generate_warning(req_credits)

    def _increase_laser_level(self):
        if self.player_ship.at_max_laser_level:
            self.warning_text = self.store_font.render(f'Laser level at maximum.', 1, RED)
            self.display_warning = True
            return

        req_credits = int(STORE_LASER_LEVEL_UPGRADE_COST * (1 + self.player_ship.laser_level))
        if self.credits > req_credits:
            self.display_warning = False
            self.player_ship.laser_level += 1
            self.credits -= req_credits

        else:
            self._generate_warning(req_credits)

    def _increase_lives(self, level):
        req_credits = STORE_LIVES_COST + STORE_LIVES_PER_LEVEL_ADDITIONAL_COST * (level - 1)
        if self.credits > req_credits:
            self.lives += 1
            self.credits -= req_credits

        else:
            self._generate_warning(req_credits)
        
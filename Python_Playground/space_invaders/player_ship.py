import pygame
from ship import Ship
import os
from settings import (WIDTH, HEIGHT, FPS, UPGRADE_PERCENT, YELLOW, GREEN, RED, PLAYER_MIN_FIRE_RATE, PLAYER_STARTING_HEALTH,PLAYER_STARTING_DMG,
                      PLAYER_LASER_STARTING_REGEN, PLAYER_STARTING_MOVEMENT_SPEED, PLAYER_LASER_BASE_CHARGE_LEVEL, PLAYER_LASER_BASE_COST,
                      PLAYER_INVINSIBLE_AFTER_DEATH_PERIOD, PLAYER_SHIELD_REGEN_BASE, PLAYER_SHIELD_STRENGTH_PER_LEVEL, MAX_SHIELD_LEVEL)


class PlayerSpaceShip(Ship):
    def __init__(self):
        super().__init__()
        self.image = pygame.transform.rotate(pygame.image.load(os.path.join(r'Python_Playground\space_invaders\Assets', 'spaceship_yellow.png')), 180).convert_alpha()
        self.image = pygame.transform.scale(self.image, self.ship_size)
        self.mask = pygame.mask.from_surface(self.image)
        x = WIDTH // 2 - self.image.get_width() // 2
        y = HEIGHT - self.image.get_height() * 2
        self.rect = pygame.Rect(x, y, self.image.get_width(), self.image.get_height())
        self.laser_image = pygame.image.load(os.path.join(r'Python_Playground\space_invaders\Assets', 'red_laser.png')).convert_alpha()

        self.health_upgrades = 0
        self.damage_upgrades = 0
        self.laser_regen_upgrades = 0
        self.speed_upgrades = 0
        self.laser_max_charge_upgrades = 0
        self.laser_cost_upgrades = 0
        self.shield_regen_upgrades = 0
        self.shield_cooldown_upgrades = 0

        self.laser_charge = self.laser_max_charge
        self.health = self.max_health
        self.laser_timer = PLAYER_MIN_FIRE_RATE * FPS / 60
        self.invinsible_timer = 0

    @property
    def shield_regen(self):
        return PLAYER_SHIELD_REGEN_BASE * self.improvment_multiplyer(self.shield_regen_upgrades) / FPS

    @property
    def max_shield_strength(self):
        return PLAYER_SHIELD_STRENGTH_PER_LEVEL * self.shield_level

    @property
    def shield_cooldown_modifier(self):
        return self.improvment_multiplyer(self.shield_cooldown_upgrades)

    @property
    def max_health(self):
        return PLAYER_STARTING_HEALTH * self.improvment_multiplyer(self.health_upgrades)

    @property
    def base_damage(self):
        return PLAYER_STARTING_DMG * self.improvment_multiplyer(self.damage_upgrades)    

    @property
    def laser_regen(self):
        return PLAYER_LASER_STARTING_REGEN * self.improvment_multiplyer(self.laser_regen_upgrades)  

    @property
    def speed(self):
        return PLAYER_STARTING_MOVEMENT_SPEED * self.improvment_multiplyer(self.speed_upgrades)

    @property
    def laser_max_charge(self):
        return PLAYER_LASER_BASE_CHARGE_LEVEL * self.improvment_multiplyer(self.laser_max_charge_upgrades)

    @property
    def laser_cost(self):
        return self.laser_type_cost_multipliers[self.laser_level] * PLAYER_LASER_BASE_COST / self.improvment_multiplyer(self.laser_cost_upgrades)

    @property
    def is_invinsible(self):
        return self.invinsible_timer

    @property
    def can_fire(self):
        return self.laser_charge >= self.laser_cost and self.laser_timer >= PLAYER_MIN_FIRE_RATE * (PLAYER_LASER_STARTING_REGEN / self.laser_regen) * self.laser_type_fire_rate_multipliers[self.laser_level]

    @property
    def at_max_shield_level(self):
        return self.shield_level >= MAX_SHIELD_LEVEL

    @property
    def at_max_laser_level(self):
        return self.laser_level >= len(self.laser_types) - 1

    def improvment_multiplyer(self, num_upgrades):
        return 1 + UPGRADE_PERCENT * num_upgrades

    def draw(self, window):
        super().draw(window)

        # Laser charge bar
        pygame.draw.rect(window, YELLOW, pygame.Rect(self.rect.x - 10,
                                                          self.rect.y + (self.ship_size[0] * (self.laser_max_charge - self.laser_charge) // self.laser_max_charge),
                                                          5,
                                                          (self.ship_size[0] * (self.laser_charge) // self.laser_max_charge)))

        # Health bar
        if self.health > 0.5 * self.max_health:
            color = GREEN
        elif self.health > 0.25 * self.max_health:
            color = YELLOW
        else:
            color = RED

        pygame.draw.rect(window, color, pygame.Rect(self.rect.x,
                                                          self.rect.y + self.ship_size[1] + 10,
                                                          self.ship_size[0] * (self.health / self.max_health),
                                                          5))

    def update(self):
        if self.shield_level:
            self.update_shield()

        self.laser_timer += 1
        self.laser_charge = min(self.laser_charge + self.laser_regen / FPS, self.laser_max_charge)
        self.invinsible_timer = max(self.invinsible_timer - 1, 0)
        if self.invinsible_timer:
            self.draw_invincibility()

    def draw_invincibility(self):
        if (self.invinsible_timer // (FPS / 2)) % 2:
            self.image.set_alpha(128)
        else:
            self.image.set_alpha(255)

    def fire(self):
        if self.can_fire:
            self.laser_charge -= self.laser_cost
            self.laser_timer = 0
            return self.laser_types[self.laser_level]()

    def take_hit(self, damage):
        if not self.is_invinsible:
            self.health -= damage

    def player_up(self):
        if self.rect.y > 0 + self.speed // FPS:
            self.rect.y -= self.speed //FPS
        else:
            self.rect.y = 0

    def player_down(self):
        if self.rect.y + self.image.get_height() < HEIGHT - self.speed // FPS:
            self.rect.y += self.speed // FPS
        else:
            self.rect.y = HEIGHT - self.image.get_height()

    def player_left(self):
        if self.rect.x > 0 + self.speed // FPS:
            self.rect.x -= self.speed // FPS
        else:
            self.rect.x = 0

    def player_right(self):
        if self.rect.x + self.image.get_width() < WIDTH - self.speed // FPS:
            self.rect.x += self.speed // FPS
        else:
            self.rect.x = WIDTH - self.image.get_width()

    def post_death_actions(self):
        self.health = self.max_health
        self.invinsible_timer = PLAYER_INVINSIBLE_AFTER_DEATH_PERIOD * FPS / 60

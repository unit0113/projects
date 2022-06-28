import pygame
import os
import random
import time


WIDTH, HEIGHT = 1200, 1000
FPS = 60
MOVEMENT_SPEED = 480
LASER_SPEED = 960
AI_LASER_SPEED = 720
LASER_SIZE = (6, 30)
LASER_REGEN = 100
LASER_COST = 60
AI_BASE_FIRE_RATE = 5
AI_BASE_SPAWN_RATE = 10
AI_BASE_SPEED = 120
AI_BASE_DMG = 15
AI_BASE_HEALTH = 100
PLAYER_LIVES = 5
PLAYER_STARTING_HEALTH = 100
PLAYER_STARTING_DMG = 100
SHIP_SIZE = (50, 50)

WHITE = (255, 255, 255)
RED = (199, 14, 32)
YELLOW = (200, 200, 15)
GREEN = (34,139,34)
LASER_GREEN = (160, 252, 36)


class PlayerSpaceShip:
    def __init__(self, window):
        self.window = window
        self.image = pygame.transform.rotate(pygame.image.load(os.path.join(r'Python_Playground\space_invaders\Assets', 'spaceship_yellow.png')), 180)
        self.image = pygame.transform.scale(self.image, SHIP_SIZE)
        x = WIDTH // 2 - self.image.get_width() // 2
        y = HEIGHT - self.image.get_height() * 2
        self.rect = pygame.Rect(x, y, self.image.get_width(), self.image.get_height())
        self.lasers = []
        self.max_laser_charge = 100
        self.laser_charge = self.max_laser_charge
        self.max_health = PLAYER_STARTING_HEALTH
        self.health = self.max_health
        self.max_damage = PLAYER_STARTING_DMG

    @property
    def damage(self):
        return self.max_damage * random.uniform(0.5, 1.5)

    def draw(self):
        self.update_lasers()
        for laser in self.lasers:
            pygame.draw.rect(self.window, RED, laser)

        self.window.blit(self.image, (self.rect.x, self.rect.y))

        # Laser charge bar
        pygame.draw.rect(self.window, YELLOW, pygame.Rect(self.rect.x - 10,
                                                          self.rect.y + (SHIP_SIZE[0] * (100 - self.laser_charge) // 100),
                                                          5,
                                                          (SHIP_SIZE[0] * (self.laser_charge) // 100)))

        # Health bar
        if self.health > 0.5 * self.max_health:
            color = GREEN
        elif self.health > 0.25 * self.max_health:
            color = YELLOW
        else:
            color = RED

        pygame.draw.rect(self.window, color, pygame.Rect(self.rect.x,
                                                          self.rect.y + SHIP_SIZE[1] + 10,
                                                          SHIP_SIZE[0] * (self.health / self.max_health),
                                                          5))

    def update_lasers(self):
        self.laser_charge = min(self.laser_charge + LASER_REGEN / FPS, self.max_laser_charge)

        for laser in self.lasers[:]:
            if laser.y < -LASER_SIZE[1]:
                self.lasers.remove(laser)
            else:
                laser.y -= LASER_SPEED // FPS

    def is_dead(self):
        if self.health <= 0:
            return True
        
        return False

    def fire(self):
        if self.laser_charge >= 100:
            laser = pygame.Rect(self.rect.x + self.image.get_width() // 2 - LASER_SIZE[0] // 2 // 2, self.rect.y, LASER_SIZE[0], LASER_SIZE[1])
            self.lasers.append(laser)
            self.laser_charge -= LASER_COST

    def take_hit(self, damage):
        self.health -= damage


class EvilSpaceShip:
    def __init__(self, window, level):
        self.window = window
        self.level = level
        self.image = pygame.transform.scale(pygame.image.load(os.path.join(r'Python_Playground\space_invaders\Assets', 'spaceship_red.png')), SHIP_SIZE)
        self.rect = pygame.Rect(random.randint(20, WIDTH - 20 - SHIP_SIZE[0]), -10 - SHIP_SIZE[1], self.image.get_width(), self.image.get_height())
        self.lasers = []
        self.max_health = int(AI_BASE_HEALTH * random.uniform(0.8, 1.2) * (1 + (self.level - 1) / 10))
        self.health = self.max_health
        self.speed = int(AI_BASE_SPEED * random.uniform(0.8, 1.2) * (1 + (self.level - 1) / 10))

    def draw(self):
        self.rect.y += self.speed // FPS

        for laser in self.lasers:
            pygame.draw.rect(self.window, LASER_GREEN, laser)

        self.window.blit(self.image, (self.rect.x, self.rect.y))

    def is_dead(self):
        if self.health <= 0:
            return True

        return False

    def fire(self):
        if self.rect.y > 0 and random.uniform(0, 10) < AI_BASE_FIRE_RATE * (1 + (self.level - 1) / 10) / FPS:
            laser = pygame.Rect(self.rect.x + self.image.get_width() // 2 - LASER_SIZE[0] // 2 // 2, self.rect.y, LASER_SIZE[0], LASER_SIZE[1])
            return laser

    def take_hit(self, damage):
        self.health -= damage


class BadGuyManager:
    def __init__(self, window, level):
        self.window = window
        self.level = level
        self.bad_guys = []
        self.evil_lasers = []

    @property
    def damage(self):
        return AI_BASE_DMG * random.uniform(0.5, 1.5) * (1 + (self.level - 1) / 10)

    def update(self):
        self.add_baddies()
        self.update_baddies()
        self.update_lasers()
        self.draw()

    def add_baddies(self):
        if random.uniform(0, 10) < AI_BASE_SPAWN_RATE * (1 + (self.level - 1) / 10) / FPS:
            self.bad_guys.append(EvilSpaceShip(self.window, self.level))

    def update_baddies(self):
        for baddie in self.bad_guys[:]:
            baddie.rect.y += baddie.speed // FPS
            if baddie.rect.y > HEIGHT:
                self.bad_guys.remove(baddie)
                continue

            check_fire = baddie.fire()
            if check_fire:
                self.evil_lasers.append(check_fire)

    def update_lasers(self):
        for laser in self.evil_lasers[:]:
            laser.y += AI_LASER_SPEED // FPS
            if laser.y > HEIGHT:
                self.evil_lasers.remove(laser)

    def draw(self):
        for laser in self.evil_lasers:
            pygame.draw.rect(self.window, LASER_GREEN, laser)

        for baddie in self.bad_guys:
            self.window.blit(baddie.image, (baddie.rect.x, baddie.rect.y))


class SpaceGame:
    def __init__(self, window):
        self.window = window
        self.background = pygame.transform.scale(pygame.image.load(os.path.join(r'Python_Playground\space_invaders\Assets', 'space.png')), (WIDTH, HEIGHT))
        self.player = PlayerSpaceShip(self.window)
        self.player_lives = PLAYER_LIVES
        self.level = 1
        self.bad_guy_manager = BadGuyManager(self.window, self.level)
        self.labels_font = pygame.font.SysFont('verdana', 30, bold=True)
        self.bad_guys = []
        self.score = 0

    def player_up(self):
        if self.player.rect.y > 0 + MOVEMENT_SPEED // FPS:
            self.player.rect.y -= MOVEMENT_SPEED //FPS

    def player_down(self):
        if self.player.rect.y + self.player.image.get_height() < HEIGHT - MOVEMENT_SPEED // FPS:
            self.player.rect.y += MOVEMENT_SPEED // FPS

    def player_left(self):
        if self.player.rect.x > 0 + MOVEMENT_SPEED // FPS:
            self.player.rect.x -= MOVEMENT_SPEED // FPS

    def player_right(self):
        if self.player.rect.x + self.player.image.get_width() < WIDTH - MOVEMENT_SPEED // FPS:
            self.player.rect.x += MOVEMENT_SPEED // FPS

    def player_fire(self):
        self.player.fire()

    def draw(self):
        self.window.blit(self.background, (0, 0))
        self.bad_guy_manager.update()
        self.player.draw()
        self.check_hits()
        self.check_player_death()

        lives_label = self.labels_font.render(f'Lives: {self.player_lives}', 1, WHITE)
        self.window.blit(lives_label, (15, 15))
        level_label = self.labels_font.render(f'Level: {self.level}', 1, WHITE)
        self.window.blit(level_label, (WIDTH - level_label.get_width() - 15, 15))
        score_label = self.labels_font.render(f'Score: {self.score}', 1, WHITE)
        self.window.blit(score_label, (WIDTH // 2 - score_label.get_width() // 2, 15))

        pygame.display.update()

    def check_hits(self):
        # Check laser hits on bad guys
        for laser in self.player.lasers[:]:
            for baddie in self.bad_guy_manager.bad_guys[:]:
                if laser.colliderect(baddie):
                    baddie.take_hit(self.player.damage)
                    if laser in self.player.lasers: self.player.lasers.remove(laser)
                    if baddie.is_dead():
                        self.score += baddie.max_health * 10
                        self.bad_guy_manager.bad_guys.remove(baddie)

        # Check ship-to-ship collision
        for baddie in self.bad_guy_manager.bad_guys[:]:
            if self.player.rect.colliderect(baddie):
                self.player.take_hit(baddie.max_health)
                self.score += baddie.max_health * 10
                self.bad_guy_manager.bad_guys.remove(baddie)

        # Check laser hits on player
        for laser in self.bad_guy_manager.evil_lasers[:]:
            if laser.colliderect(self.player.rect):
                self.player.take_hit(self.bad_guy_manager.damage)
                if laser in self.bad_guy_manager.evil_lasers:
                    self.bad_guy_manager.evil_lasers.remove(laser)

    def check_player_death(self):
        if self.player.is_dead():
            if self.player_lives > 0:
                self.player_lives -= 1
                self.player.health = self.player.max_health
            else:
                game_over(self.score, self.level)
                        

def initialize_pygame():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Space Invaders")
    pygame.font.init()

    return window


def game_over(score, level):
    pass


def main():
    window = initialize_pygame()
    space_game = SpaceGame(window)
    clock = pygame.time.Clock()

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_q]:
            pygame.quit()
            quit()

        if keys[pygame.K_r]:
            main()

        if keys[pygame.K_UP]:
            space_game.player_up()

        if keys[pygame.K_DOWN]:
            space_game.player_down()

        if keys[pygame.K_LEFT]:
            space_game.player_left()

        if keys[pygame.K_RIGHT]:
            space_game.player_right()
        
        if keys[pygame.K_SPACE]:
            space_game.player_fire()

        space_game.draw()


if __name__ == "__main__":
    main()
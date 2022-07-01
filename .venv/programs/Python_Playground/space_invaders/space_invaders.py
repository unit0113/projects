import pygame
import os
import random


WIDTH, HEIGHT = 1200, 1000
FPS = 60

LASER_SPEED = 960
AI_LASER_SPEED = 720
LASER_SIZE = (6, 30)
LASER_STARTING_REGEN = 100
LASER_BASE_COST = 60
LASER_BASE_CHARGE_LEVEL = 100

AI_BASE_FIRE_CHANCE = 5
AI_BASE_FIRE_RATE = 25
AI_BASE_SPAWN_RATE = 10
AI_BASE_SPEED = 120
AI_BASE_DMG = 20
AI_BASE_HEALTH = 100

PLAYER_LIVES = 5
PLAYER_STARTING_MOVEMENT_SPEED = 480
PLAYER_STARTING_HEALTH = 100
PLAYER_STARTING_DMG = 100
PLAYER_MIN_FIRE_RATE = 25
PLAYER_INVINSIBLE_AFTER_DEATH_PERIOD = 120

SHIP_SIZE = (50, 50)
LEVEL_DURATION = 20

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

BONUS_NO_DAMAGE_TAKEN = 0.1
BONUS_ALL_ENEMIES_KILLED = 0.1

WHITE = (255, 255, 255)
RED = (199, 14, 32)
YELLOW = (200, 200, 15)
GREEN = (34,139,34)
LASER_GREEN = (160, 252, 36)
STEEL = (67, 70, 75)


"""TODO:
shields
power-ups
shop
different ship types
different enemy types, use a factory
mouse controls
fix alignment on scorecard

"""


class Laser:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, *LASER_SIZE)
        self.mask = pygame.mask.Mask(LASER_SIZE, True)

    def draw(self, window):
        pygame.draw.rect(window, RED, self.rect)


class PlayerSpaceShip:
    def __init__(self, window):
        pygame.sprite.Sprite.__init__(self)
        self.window = window
        self.image = pygame.transform.rotate(pygame.image.load(os.path.join(r'Python_Playground\space_invaders\Assets', 'spaceship_yellow.png')), 180).convert_alpha()
        self.image = pygame.transform.scale(self.image, SHIP_SIZE)
        self.mask = pygame.mask.from_surface(self.image)
        x = WIDTH // 2 - self.image.get_width() // 2
        y = HEIGHT - self.image.get_height() * 2
        self.rect = pygame.Rect(x, y, self.image.get_width(), self.image.get_height())
        self.laser_types = [self.laser1, self.laser2, self.laser3]
        self.laser_type_dmg_multipliers = [1, 0.65, 0.5]
        self.laser_type_current_index = 0
        self.lasers = []

        self.health_upgrades = 0
        self.damage_upgrades = 0
        self.laser_regen_upgrades = 0
        self.speed_upgrades = 0
        self.laser_max_charge_upgrades = 0
        self.laser_cost_upgrades = 0

        self.laser_charge = self.laser_max_charge
        self.health = self.max_health
        self.laser_timer = PLAYER_MIN_FIRE_RATE * FPS / 60
        self.invinsible_timer = 0

    @property
    def max_health(self):
        return PLAYER_STARTING_HEALTH * self.improvment_multiplyer(self.health_upgrades)

    @property
    def max_damage(self):
        return PLAYER_STARTING_DMG * self.improvment_multiplyer(self.damage_upgrades)    

    @property
    def laser_regen(self):
        return LASER_STARTING_REGEN * self.improvment_multiplyer(self.laser_regen_upgrades)  

    @property
    def speed(self):
        return PLAYER_STARTING_MOVEMENT_SPEED * self.improvment_multiplyer(self.speed_upgrades)

    @property
    def laser_max_charge(self):
        return LASER_BASE_CHARGE_LEVEL * self.improvment_multiplyer(self.laser_max_charge_upgrades)

    @property
    def laser_cost(self):
        return LASER_BASE_COST * self.improvment_multiplyer(self.laser_cost_upgrades)

    @property
    def damage(self):
        return self.max_damage * random.uniform(0.5, 1.5) * self.laser_type_dmg_multipliers[self.laser_type_current_index]

    @property
    def is_invinsible(self):
        return self.invinsible_timer

    @property
    def is_dead(self):
        return self.health <= 0

    @property
    def can_fire(self):
        return self.laser_charge >= self.laser_cost and self.laser_timer > PLAYER_MIN_FIRE_RATE

    def improvment_multiplyer(self, num_upgrades):
        return 1 + num_upgrades * UPGRADE_PERCENT

    def draw(self):
        self.update()
        for laser in self.lasers:
            laser.draw(self.window)

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

    def update(self):
        self.update_lasers()
        self.laser_timer += 1
        self.invinsible_timer = max(self.invinsible_timer - 1, 0)
        if self.invinsible_timer:
            self.draw_invincibility()

    def update_lasers(self):
        self.laser_charge = min(self.laser_charge + self.laser_regen / FPS, self.laser_max_charge)

        for laser in self.lasers[:]:
            if laser.rect.y < -LASER_SIZE[1]:
                self.lasers.remove(laser)
            else:
                laser.rect.y -= LASER_SPEED // FPS

    def draw_invincibility(self):
        if (self.invinsible_timer // (FPS / 2)) % 2:
            self.image.set_alpha(128)
        else:
            self.image.set_alpha(255)

    def fire(self):
        if self.can_fire:
            self.lasers += self.laser_types[self.laser_type_current_index]()
            self.laser_charge -= self.laser_cost
            self.laser_timer = 0

    def laser1(self):
        laser = Laser(self.rect.x + self.image.get_width() // 2 - LASER_SIZE[0] // 2, self.rect.y)
        return [laser]

    def laser2(self):
        laser1 = Laser(self.rect.x, self.rect.y + 15)
        laser2 = Laser(self.rect.x + SHIP_SIZE[0] - LASER_SIZE[0], self.rect.y + 15)
        return [laser1, laser2]

    def laser3(self):
        laser1 = Laser(self.rect.x + self.image.get_width() // 2 - LASER_SIZE[0] // 2, self.rect.y)
        laser2 = Laser(self.rect.x, self.rect.y + 15)
        laser3 = Laser(self.rect.x + SHIP_SIZE[0] - LASER_SIZE[0], self.rect.y + 15)
        
        return [laser1, laser2, laser3]

    def take_hit(self, damage):
        if not self.is_invinsible:
            self.health -= damage

    def player_up(self):
        if self.rect.y > 0 + self.speed // FPS:
            self.rect.y -= self.speed //FPS

    def player_down(self):
        if self.rect.y + self.image.get_height() < HEIGHT - self.speed // FPS:
            self.rect.y += self.speed // FPS

    def player_left(self):
        if self.rect.x > 0 + self.speed // FPS:
            self.rect.x -= self.speed // FPS

    def player_right(self):
        if self.rect.x + self.image.get_width() < WIDTH - self.speed // FPS:
            self.rect.x += self.speed // FPS


class EvilSpaceShip:
    def __init__(self, window, level):
        pygame.sprite.Sprite.__init__(self)
        self.window = window
        self.level = level
        self.image = pygame.transform.scale(pygame.image.load(os.path.join(r'Python_Playground\space_invaders\Assets', 'spaceship_red.png')), SHIP_SIZE).convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = pygame.Rect(random.randint(20, WIDTH - 20 - SHIP_SIZE[0]), -10 - SHIP_SIZE[1], self.image.get_width(), self.image.get_height())
        self.max_health = int(AI_BASE_HEALTH * random.uniform(0.8, 1.2) * (1 + (self.level - 1) / 10))
        self.health = self.max_health
        self.speed = int(AI_BASE_SPEED * random.uniform(0.8, 1.2) * (1 + (self.level - 1) / 10))
        self.laser_timer = AI_BASE_FIRE_RATE * FPS / 60

    @property
    def is_dead(self):
        return self.health <= 0

    @property
    def can_fire(self):
        return self.rect.y > 0 and self.laser_timer > AI_BASE_FIRE_RATE

    def draw(self):
        self.laser_timer += 1
        self.window.blit(self.image, (self.rect.x, self.rect.y))

    def fire(self):
        if self.can_fire and random.uniform(0, 10) < AI_BASE_FIRE_CHANCE * (1 + (self.level - 1) / 10) / FPS:
            self.laser_timer = 0
            return Laser(self.rect.x + self.image.get_width() // 2 - LASER_SIZE[0] // 2 // 2, self.rect.y)

    def take_hit(self, damage):
        self.health -= damage


class BadGuyManager:
    def __init__(self, window, level):
        self.window = window
        self.level = level
        self.bad_guys = []
        self.evil_lasers = []
        self.no_baddies_escaped = True
        self.new_baddies_generation = True

    @property
    def damage(self):
        return AI_BASE_DMG * random.uniform(0.5, 1.5) * (1 + (self.level - 1) / 10)

    def update(self):
        self.add_baddies()
        self.update_baddies()
        self.update_lasers()
        self.draw()

    def add_baddies(self):
        if self.new_baddies_generation and random.uniform(0, 10) < AI_BASE_SPAWN_RATE * (1 + (self.level - 1) / 10) / FPS:
            self.bad_guys.append(EvilSpaceShip(self.window, self.level))

    def update_baddies(self):
        for baddie in self.bad_guys[:]:
            baddie.rect.y += baddie.speed // FPS
            if baddie.rect.y > HEIGHT:
                self.bad_guys.remove(baddie)
                self.no_baddies_escaped = False
                continue

            check_fire = baddie.fire()
            if check_fire:
                self.evil_lasers.append(check_fire)

    def update_lasers(self):
        for laser in self.evil_lasers[:]:
            laser.rect.y += AI_LASER_SPEED // FPS
            if laser.rect.y > HEIGHT:
                self.evil_lasers.remove(laser)

    def draw(self):
        for laser in self.evil_lasers:
            laser.draw(self.window)

        for baddie in self.bad_guys:
            baddie.draw()


class SpaceGame:
    def __init__(self, window):
        self.window = window
        self.background = pygame.transform.scale(pygame.image.load(os.path.join(r'Python_Playground\space_invaders\Assets', 'space.png')), (WIDTH, HEIGHT))
        self.player = PlayerSpaceShip(self.window)
        self.player_lives = PLAYER_LIVES
        self.level = 0
        self.bad_guy_manager = BadGuyManager(self.window, self.level)
        self.labels_font = pygame.font.SysFont('verdana', 30, bold=True)
        self.scorecard_font = pygame.font.SysFont('verdana', 25, bold=True)
        self.store_title_font = pygame.font.SysFont('verdana', 40, bold=True)
        self.score = 0
        self.credits = 0
        self.max_level_duration = LEVEL_DURATION * FPS
        self.initialize_round()

    @property
    def is_round_end(self):
        return self.level_duration > self.max_level_duration and not self.bad_guy_manager.bad_guys

    def initialize_round(self):
        self.player.health = self.player.max_health
        self.level += 1
        self.level_duration = 0
        self.no_damage_taken = True
        self.bad_guy_manager.no_baddies_escaped = True
        self.bad_guy_manager.new_baddies_generation = True
        self.num_baddies_killed_round = 0
        self.score_start_of_round = self.score
        self.bad_guy_manager.bad_guys.clear()
        self.bad_guy_manager.evil_lasers.clear()
        self.draw()
        self.open_store() # Remove when done with store

    def end_round(self):
        self.round_scoreboard()
        self.open_store()
        self.initialize_round()

    def round_scoreboard(self):
        self.window.blit(self.background, (0, 0))
        self.player.draw()

        spacer = 0
        enemies_text = self.scorecard_font.render(f'{"Enemy ships destroyed":.<30}{self.num_baddies_killed_round:.>10}', 1, WHITE)
        self.window.blit(enemies_text, (WIDTH // 2 - enemies_text.get_width() // 2, HEIGHT // 3 - enemies_text.get_height() // 2))

        spacer += 1
        round_points = self.score - self.score_start_of_round
        points_text = self.scorecard_font.render(f'{"Points earned:":.<30}{round_points:.>10}', 1, WHITE)
        self.window.blit(points_text, (WIDTH // 2 - points_text.get_width() // 2, HEIGHT // 3 + 30 * spacer - points_text.get_height() // 2))

        spacer += 1
        level_bonus_pts = int(round_points * (self.level / 10))
        level_bonus_text = self.scorecard_font.render(f'{"Level bonus:":.<30}{level_bonus_pts:.>10}', 1, WHITE)
        self.window.blit(level_bonus_text, (WIDTH // 2 - level_bonus_text.get_width() // 2, HEIGHT // 3 + 30 * spacer - level_bonus_text.get_height() // 2))

        no_damamge_pts = all_baddies_killed_pts = 0
        if self.no_damage_taken:
            spacer += 1
            no_damamge_pts = int(round_points * BONUS_NO_DAMAGE_TAKEN)
            no_damage_text = self.scorecard_font.render(f'{"No damage taken bonus:":.<30}{no_damamge_pts:.>10}', 1, WHITE)
            self.window.blit(no_damage_text, (WIDTH // 2 - no_damage_text.get_width() // 2, HEIGHT // 3 + 30 * spacer - no_damage_text.get_height() // 2))

        if self.bad_guy_manager.no_baddies_escaped:
            spacer += 1
            all_baddies_killed_pts = int(round_points * BONUS_ALL_ENEMIES_KILLED)
            all_killed_text = self.scorecard_font.render(f'{"Total destruction bonus:":.<30}{all_baddies_killed_pts:.>10}', 1, WHITE)
            self.window.blit(all_killed_text, (WIDTH // 2 - all_killed_text.get_width() // 2, HEIGHT // 3 + 30 * spacer - all_killed_text.get_height() // 2))

        spacer += 2
        self.score = self.score_start_of_round + round_points + level_bonus_pts + no_damamge_pts + all_baddies_killed_pts
        total_pts_text = self.scorecard_font.render(f'{"Total score:":.<30}{self.score:.>10}', 1, WHITE)
        self.window.blit(total_pts_text, (WIDTH // 2 - total_pts_text.get_width() // 2, HEIGHT // 3 + 30 * spacer - total_pts_text.get_height() // 2))

        spacer += 3
        instructions_text = self.labels_font.render('Press C to continue.', 1, WHITE)
        self.window.blit(instructions_text, (WIDTH // 2 - instructions_text.get_width() // 2, HEIGHT // 3 + 30 * spacer - instructions_text.get_height() // 2))

        pygame.display.update()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    return

    def open_store(self):
        self.window.blit(self.background, (0, 0))
        store_rect = pygame.Rect(STORE_WINDOW_PADDING, STORE_WINDOW_PADDING, WIDTH - 2 * STORE_WINDOW_PADDING, HEIGHT - 2 * STORE_WINDOW_PADDING)
        pygame.draw.rect(self.window, STEEL, store_rect, border_radius=10)

        store_title_text = self.store_title_font.render(f'{"Upgrade Store"}', 1, WHITE)
        self.window.blit(store_title_text, (WIDTH // 2 - store_title_text.get_width() // 2, STORE_WINDOW_PADDING + store_title_text.get_height() // 2))








        pygame.display.update()
        pygame.time.delay(5000)







        

    def player_up(self):
        self.player.player_up()

    def player_down(self):
        self.player.player_down()

    def player_left(self):
        self.player.player_left()

    def player_right(self):
        self.player.player_right()

    def player_fire(self):
        self.player.fire()

    def draw(self):
        self.level_duration += 1
        if self.level_duration > self.max_level_duration:
            self.bad_guy_manager.new_baddies_generation = False
        self.window.blit(self.background, (0, 0))
        self.bad_guy_manager.update()
        self.player.draw()
        self.check_hits()
        if self.is_round_end:
            self.end_round()
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
                if laser.mask.overlap(baddie.mask, (baddie.rect.x - laser.rect.x, baddie.rect.y - laser.rect.y)):
                    baddie.take_hit(self.player.damage)
                    if laser in self.player.lasers: self.player.lasers.remove(laser)
                    if baddie.is_dead:
                        self.score += baddie.max_health * 10
                        self.bad_guy_manager.bad_guys.remove(baddie)
                        self.num_baddies_killed_round += 1

        # Check for player damage
        if not self.player.is_invinsible:
            # Check ship-to-ship collision
            for baddie in self.bad_guy_manager.bad_guys[:]:
                if self.player.mask.overlap(baddie.mask, (self.player.rect.x - baddie.rect.x, self.player.rect.y - baddie.rect.y)):
                    self.player.take_hit(baddie.max_health)
                    self.score += baddie.max_health * 10
                    self.bad_guy_manager.bad_guys.remove(baddie)
                    self.no_damage_taken = False

            # Check laser hits on player
            for laser in self.bad_guy_manager.evil_lasers[:]:
                if laser.mask.overlap(self.player.mask, (self.player.rect.x - laser.rect.x, self.player    .rect.y - laser.rect.y)):
                    self.player.take_hit(self.bad_guy_manager.damage)
                    if laser in self.bad_guy_manager.evil_lasers:
                        self.bad_guy_manager.evil_lasers.remove(laser)
                        self.no_damage_taken = False

    def check_player_death(self):
        if self.player.is_dead:
            if self.player_lives > 0:
                self.player_lives -= 1
                self.player.health = self.player.max_health
                self.player.invinsible_timer = PLAYER_INVINSIBLE_AFTER_DEATH_PERIOD * FPS/ 60
            else:
                self.game_over()
    
    def game_over(self):
        self.window.blit(self.background, (0, 0))
        self.game_over_font = pygame.font.SysFont('verdana', 50, bold=True)

        death_text = self.game_over_font.render(f'You Died...', 1, WHITE)
        self.window.blit(death_text, (WIDTH // 2 - death_text.get_width() // 2, HEIGHT // 4 - death_text.get_height() // 2))

        level_text = self.game_over_font.render(f'You reached level {self.level}', 1, WHITE)
        self.window.blit(level_text, (WIDTH // 2 - level_text.get_width() // 2, HEIGHT // 4 + 50 - level_text.get_height() // 2))

        score_text = self.game_over_font.render(f'Your final score is {self.score}', 1, WHITE)
        self.window.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 4 + 100 - score_text.get_height() // 2))

        instructions_text = self.game_over_font.render('Press C to play again, or press Q to quit.', 1, WHITE)
        self.window.blit(instructions_text, (WIDTH // 2 - instructions_text.get_width() // 2, HEIGHT // 4 + 150 - instructions_text.get_height() // 2))
        pygame.display.update()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    main()

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    pygame.quit()
                    quit()
                        

def initialize_pygame():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Space Invaders")
    pygame.font.init()

    return window


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

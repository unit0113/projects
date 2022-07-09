import random
from player_ship import PlayerSpaceShip
from enemy_ships import EvilSpaceShip
from settings import HEIGHT, FPS, PLAYER_LASER_SPEED, AI_LASER_SPEED, AI_BASE_SPAWN_RATE


class AssetManager:
    def __init__(self, window):
        self.window = window
        self.level = 0
        self.player = PlayerSpaceShip()

    def new_round(self):
        self.player.health = self.player.max_health
        self.player.shield_strength = self.player.max_shield_strength
        self.level +=1
        self.bad_guys = []
        self.evil_lasers = []
        self.good_lasers = []
        self.good_missiles = []
        self.num_baddies_killed_round = 0
        self.no_damage_taken = True
        self.no_baddies_escaped = True
        self.new_baddies_generation = True

    def player_fire(self):
        lasers = self.player.fire()
        if lasers:
            self.good_lasers += lasers

        missiles = self.player.fire_missile()
        if missiles:
            self.good_missiles += missiles

    def update(self):
        self.update_baddies()
        self.player.update()
        self.add_baddies()
        self.update_lasers()
        self.update_missiles()

    def update_baddies(self):
        for baddie in self.bad_guys[:]:
            baddie.update()
            if baddie.rect.y > HEIGHT:
                self.bad_guys.remove(baddie)
                self.no_baddies_escaped = False
                continue

            check_fire = baddie.fire()
            if check_fire:
                self.evil_lasers += check_fire
    
    def add_baddies(self):
        if self.new_baddies_generation and random.uniform(0, 10) < AI_BASE_SPAWN_RATE * (1 + (self.level - 1) / 10) / FPS:
            new_baddie = EvilSpaceShip(self.level)
            overlap = True
            while overlap:
                overlap = False
                for baddie in self.bad_guys[:]:
                    if new_baddie.rect.colliderect(baddie.rect):
                        new_baddie = EvilSpaceShip(self.level)
                        overlap = True
                        break

            self.bad_guys.append(new_baddie)

    def update_lasers(self):
        # Bad guy lasers
        for laser in self.evil_lasers[:]:
            laser.update(AI_LASER_SPEED // FPS)
            if laser.is_off_screen:
                self.evil_lasers.remove(laser)

        # Player lasers
        for laser in self.good_lasers[:]:
            laser.update(-PLAYER_LASER_SPEED // FPS)
            if laser.is_off_screen:
                self.good_lasers.remove(laser)
    
    def update_missiles(self):
        # Good guy missiles
        for missile in self.good_missiles:
            missile.update()

    def draw(self):
        for laser in self.evil_lasers:
            laser.draw(self.window)

        for laser in self.good_lasers:
            laser.draw(self.window)

        for missile in self.good_missiles:
            missile.draw(self.window)

        for baddie in self.bad_guys:
            baddie.draw(self.window)

        self.player.draw(self.window)

    def check_hits(self):
        score_change = 0

        # Check laser hits on bad guy's shields
        for laser in self.good_lasers[:]:
            for baddie in [baddie for baddie in self.bad_guys if baddie.shield_strength > 0]:
                if laser.rect.colliderect(baddie.shield_rect):
                    baddie.shield_take_hit(laser.damage)
                    self.good_lasers.remove(laser)
                    break

        # Check laser hits on bad guys
        for laser in self.good_lasers[:]:
            for baddie in self.bad_guys[:]:
                if laser.mask.overlap(baddie.mask, (baddie.rect.x - laser.rect.x, baddie.rect.y - laser.rect.y)):
                    baddie.take_hit(laser.damage)
                    self.good_lasers.remove(laser)
                    if baddie.is_dead:
                        score_change += baddie.point_value
                        self.bad_guys.remove(baddie)
                        self.num_baddies_killed_round += 1
                    break

        # Check for player damage
        if not self.player.is_invinsible:
            # Check ship-to-ship collision
            for baddie in self.bad_guys[:]:
                if self.player.mask.overlap(baddie.mask, (self.player.rect.x - baddie.rect.x, self.player.rect.y - baddie.rect.y)):
                    self.player.take_hit(baddie.max_health)
                    score_change += baddie.point_value
                    self.bad_guys.remove(baddie)
                    self.no_damage_taken = False

            # Check for laser hits on shields
            if self.player.shield_strength:
                for laser in self.evil_lasers[:]:
                    if laser.rect.colliderect(self.player.shield_rect):
                        self.player.shield_take_hit(laser.damage)
                        if laser in self.evil_lasers:
                            self.evil_lasers.remove(laser)

            # Check laser hits on player
            for laser in self.evil_lasers[:]:
                if laser.mask.overlap(self.player.mask, (self.player.rect.x - laser.rect.x, self.player.rect.y - laser.rect.y)):
                    self.player.take_hit(laser.damage)
                    if laser in self.evil_lasers:
                        self.evil_lasers.remove(laser)
                        self.no_damage_taken = False

        return score_change

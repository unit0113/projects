import pygame
from pygame import mixer

from settings import WIDTH, FIGHTER_HEALTH


MOVE_SPEED = 10
JUMP_SPEED = 40
GRAVITY = 4
ATTACK_1_STRENGTH = 25
ATTACK_2_STRENGTH = 10
ATTACK_1_TIMEOUT = 0.50
ATTACK_2_TIMEOUT = 0.25

FRAME_TIME = 0.05


class Fighter():
    def __init__(self, x: int, y: int, sprites: dict[str: list[pygame.surface.Surface]], offset: tuple[int, int], controls: dict, weapon_fx: pygame.mixer.Sound) -> None:
        self.facing_left = False
        self.rect = pygame.Rect((x, y, 80, 180))
        self.y_vel = 0
        self.min_y = y
        self.attack_type = 0
        self.attack_timer = 0
        self.enemy = None
        self.max_hp = FIGHTER_HEALTH
        self.curr_hp = FIGHTER_HEALTH
        self.sprites = sprites
        self.state = 'idle'
        self.frame_index = 0
        self.image = self.sprites[self.state][self.frame_index]
        self.offset = offset
        self.frame_timer = 0
        self.controls = controls
        self.weapon_fx = weapon_fx

    def set_enemy(self, target: 'Fighter'):
        self.enemy = target
        if self.enemy.rect.centerx < self.rect.centerx:
            self.facing_left = True

    def update(self, dt: float) -> None:
        # Update times
        self.attack_timer -= dt
        self.frame_timer += dt

        # Update animation frame
        if self.frame_timer > FRAME_TIME:
            self.frame_timer = 0
            self.frame_index += 1
            if self.frame_index >= len(self.sprites[self.state]):
                self.frame_index = 0
                # Return to idle state after completing animations
                if self.state == 'death':
                    self.frame_index = len(self.sprites[self.state]) - 1
                elif self.state in ['attack1', 'attack2', 'hit']:
                    self.update_state('idle')
                    self.attack_type = 0

        # Y movement
        self.rect.y = min(self.rect.y - self.y_vel, self.min_y)
        self.y_vel -= GRAVITY
        if self.on_ground:
            self.y_vel = 0
            # Return to idle state after jump
            if self.state == 'jump':
                self.update_state('idle')

    def move(self, inputs: pygame.key.ScancodeWrapper) -> None:
        # Do not update if round is over
        if self.state == 'death' or self.enemy.state == 'death':
            return

        # Disable other commands when attacking
        if not self.attack_type:
            # Move x
            dx = 0
            # Move left
            if inputs[self.controls['left']]:
                dx = -MOVE_SPEED
                self.facing_left = True
            # Move right
            if inputs[self.controls['right']]:
                dx = MOVE_SPEED
                self.facing_left = False

            # Clamp and update x
            if dx:
                self.rect.x = max(min(self.rect.x + dx, WIDTH - self.rect.width), 0)
                if self.state != 'run':
                    self.update_state('run')

            # When stopped running, reset to idle
            elif self.state == 'run':
                self.update_state('idle')

            # Move y
            if inputs[self.controls['jump']] and self.on_ground :
                self.y_vel += JUMP_SPEED
                self.update_state('jump')
            
            # Attack
            if inputs[self.controls['attack1']] and self.attack_timer < -ATTACK_1_TIMEOUT:
                self.attack_type = 1
                self.attack_timer = ATTACK_1_TIMEOUT
                self.state = 'attack1'
            elif inputs[self.controls['attack2']] and self.attack_timer < -ATTACK_2_TIMEOUT:
                self.attack_type = 2
                self.attack_timer = ATTACK_2_TIMEOUT
                self.state = 'attack2'
            
            if self.attack_type != 0:
                self.frame_index = 0
                self.attack()
    
    def attack(self) -> None:\
        # Execute attack
        attack_rect = pygame.Rect(self.rect.centerx - (2 * self.rect.width * self.facing_left), self.rect.y, 2 * self.rect.width, self.rect.height)
        if attack_rect.colliderect(self.enemy.rect):
            self.enemy.take_hit(self.attack_type)
        # Play sound effects
        self.weapon_fx.play()

    def update_state(self, state: str) -> None:
        self.state = state
        self.frame_index = 0
        
    def take_hit(self, attack_type: int) -> None:
        if attack_type == 1:
            self.curr_hp = max(0, self.curr_hp - ATTACK_1_STRENGTH)
        elif attack_type == 2:
            self.curr_hp = max(0, self.curr_hp - ATTACK_2_STRENGTH)

        if self.curr_hp == 0:
            self.update_state('death')
        else:
            self.update_state('hit')

    @property
    def on_ground(self) -> bool:
        return self.rect.y == self.min_y
    
    @ property
    def game_over(self) -> bool:
        return self.curr_hp == 0 and self.frame_index == len(self.sprites[self.state]) - 1

    def draw(self, window: pygame.surface.Surface) -> None:
        img = pygame.transform.flip(self.sprites[self.state][self.frame_index], self.facing_left, False)
        window.blit(img, (self.rect.x - self.offset[0], self.rect.y - self.offset[1]))
        
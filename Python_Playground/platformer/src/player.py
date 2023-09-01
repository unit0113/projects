import pygame

from .settings import GRAVITY, TERMINAL_VELOCITY, HEIGHT

FRAME_TIME = 0.1
DEATH_TIME = 2.5
GHOST_MAX_HEIGHT = 200

class Player:
    def __init__(self, x: int, y: int, player_sprites: dict[int: pygame.surface.Surface], dead_img: pygame.surface.Surface, window: pygame.surface.Surface) -> None:
        self.sprites = player_sprites
        self.dead_img = dead_img
        self.rect = self.sprites[0].get_rect()
        self.rect.x = x
        self.rect.y = y
        self.window = window
        self.y_vel = 0
        self.prev_y_vel = 0
        self.face_left = False
        self.jumped = False
        self.frame_index = 0
        self.frame_timer = 0
        self.dead = False
        self.death_timer = 0

    def update(self, dt: float, inputs: pygame.key.ScancodeWrapper, tiles: list) -> None:
        # No updates if dead
        if self.dead:
            self.death_animation(dt)
            return
        
        dx = 0
        dy = 0
        # Move player
        if (inputs[pygame.K_SPACE] or inputs[pygame.K_w]) and not self.y_vel and not self.prev_y_vel:
            self.y_vel = -15
            self.jumped = True
        if inputs[pygame.K_LEFT] or inputs[pygame.K_a]:
            dx -= 5
            self.face_left = True
        if inputs[pygame.K_RIGHT] or inputs[pygame.K_d]:
            dx += 5
            self.face_left = False

        # Update frame counter if moving
        if dx:
            self.frame_timer += dt
        else:
            self.frame_index = 0
            self.frame_timer = 0

        # Update animation frame
        if self.frame_timer > FRAME_TIME:
            self.frame_timer = 0
            self.frame_index += 1
            if self.frame_index >= len(self.sprites):
                self.frame_index = 0

        # Add gravity
        self.prev_y_vel = self.y_vel
        self.y_vel += GRAVITY
        self.y_vel = min(self.y_vel, TERMINAL_VELOCITY)
        dy += self.y_vel

        # Check for collision
        for tile in tiles:
            # Break if no movement
            if not dx and not dy:
                break
            # Check vertical collision
            if tile.rect.colliderect(self.rect.x, self.rect.y + dy, self.rect.width, self.rect.height):
                # If going up
                if self.y_vel < 0:
                    dy = tile.rect.bottom - self.rect.top
                # If going down
                else:
                    dy = tile.rect.top - self.rect.bottom
                # Reset vertical velocity
                self.y_vel = 0

            # Check horizontal collision
            if tile.rect.colliderect(self.rect.x + dx, self.rect.y, self.rect.width, self.rect.height):
                dx = 0

        # Update rect coords
        self.rect.x += dx
        self.rect.y += dy

        # Prevent falling through bottom
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT
            self.y_vel = 0

    def death_animation(self, dt: float) -> None:
        # Reset variables on initial death
        if self.dead and not self.death_timer:
            self.y_vel = -1
        self.rect.y += self.y_vel
        self.death_timer += dt

        # Stop floating up if timeout or max height
        if self.death_timer > DEATH_TIME or self.rect.y < GHOST_MAX_HEIGHT:
            self.y_vel = 0

    def check_death(self, slime_group: pygame.sprite.Group, lava_group: pygame.sprite.Group, game_over: bool) -> bool:
        # If game already over, return
        if game_over:
            return game_over
        # Check collision with death dealers:
        self.dead = pygame.sprite.spritecollide(self, slime_group, False) or pygame.sprite.spritecollide(self, lava_group, False)
        return self.dead
    
    @property
    def death_animation_complete(self) -> bool:
        return self.dead and (self.death_timer > DEATH_TIME or self.rect.y < GHOST_MAX_HEIGHT)

    def draw(self) -> None:
        if not self.dead:
            img = pygame.transform.flip(self.sprites[self.frame_index], self.face_left, False)
            self.window.blit(img, (self.rect.x, self.rect.y))
        else:
            self.window.blit(self.dead_img, (self.rect.x, self.rect.y))

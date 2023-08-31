import pygame

from .settings import GRAVITY, TERMINAL_VELOCITY

FRAME_TIME = 0.1

class Player:
    def __init__(self, x: int, y: int, player_sprites: dict[int: pygame.surface.Surface], window: pygame.surface.Surface) -> None:
        self.sprites = player_sprites
        self.image = self.sprites[0]
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.window = window
        self.y_vel = 0
        self.face_left = False
        self.frame_index = 0
        self.frame_timer = 0

    def update(self, dt: float, inputs: pygame.key.ScancodeWrapper):
        dx = 0
        dy = 0

        if inputs[pygame.K_SPACE] and self.jumped == False:
            self.y_vel = -15
            self.jumped = True
        if inputs[pygame.K_SPACE] == False:
            self.jumped = False
        if inputs[pygame.K_LEFT]:
            dx -= 5
            self.face_left = True
        if inputs[pygame.K_RIGHT]:
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
        self.y_vel += 1
        if self.y_vel > 10:
            self.y_vel = 10
        dy += self.y_vel

        #check for collision

        #update player coordinates
        self.rect.x += dx
        self.rect.y += dy

        if self.rect.bottom > 1000:
            self.rect.bottom = 1000
            dy = 0

    def draw(self) -> None:
        img = pygame.transform.flip(self.sprites[self.frame_index], self.face_left, False)
        self.window.blit(img, (self.rect.x, self.rect.y))

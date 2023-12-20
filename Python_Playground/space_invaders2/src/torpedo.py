import pygame
import math

from .settings import FRAME_TIME


class Torpedo(pygame.sprite.Sprite):
    def __init__(
        self,
        sprites: pygame.Surface,
        pos: tuple[int, int],
        damages: tuple[float, float],
        speed: int,
        direction: tuple[float, float],
        is_left: bool,
    ):
        pygame.sprite.Sprite.__init__(self)
        self.shield_damage, self.ship_damage = damages
        self.speed = speed

        self.frame_index = 0
        self.animation_frames = [0, 0, 1]
        self.last_frame = pygame.time.get_ticks()
        self.sprites = sprites
        self.original_image = sprites[0]
        self.image = self.original_image
        self.rect = self.original_image.get_rect()
        self.pos = pos
        self.rect.midtop = pos
        self.mask = pygame.mask.from_surface(self.original_image)

        self.timer = 0
        self.direction = direction
        self.vel_vector = pygame.Vector2(-0.1 if is_left else 0.1, 0.25 * -direction[1])
        self.accel_vector = pygame.Vector2(0, 0)
        self.state = 0

    def update(self, dt: float, enemies: pygame.sprite.Group):
        """Update game object in game loop

        Args:
            dt (float): time since last frame
            enemies (pygame.sprite.Group): enemies currently in game
        """

        # Launch profile state machine
        self.timer += dt
        if self.state == 0 and self.timer > 0.5:
            self.vel_vector.x = 0
            self.accel_vector.y = 0.01 * self.direction[1]
            self.state = 1
        elif self.state == 1 and self.vel_vector.magnitude() > 1:
            self.vel_vector.x, self.vel_vector.y = self.direction[0], self.direction[1]
            self.accel_vector.y = 0
            self.state = 2
            self.animation_frames = [2, 2, 1]

        self.animate()

        # Update position and velocity
        self.pos += (
            self.vel_vector * self.speed * dt
            + 0.5 * self.accel_vector * dt * dt * self.speed
        )
        self.rect.center = self.pos
        self.vel_vector += self.accel_vector

    def animate(self) -> None:
        """Controls sprite animation of missile"""

        if pygame.time.get_ticks() > self.last_frame + FRAME_TIME:
            # Reset frame counter and increment frame
            self.last_frame = pygame.time.get_ticks()
            self.frame_index += 1
            if self.frame_index >= len(self.animation_frames):
                # Loop animation
                self.frame_index = 0
            self.original_image = self.sprites[self.animation_frames[self.frame_index]]

    def get_shield_damage(self) -> float:
        """Getter for shield damage

        Returns:
            float: damage done to shields
        """
        self.kill()
        return self.shield_damage

    def get_ship_damage(self) -> float:
        """Getter for ship damage

        Returns:
            float: damage done to ships
        """
        self.kill()
        return self.ship_damage

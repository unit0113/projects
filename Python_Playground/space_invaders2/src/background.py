import pygame

from .settings import WIDTH, HEIGHT


class Background:
    def __init__(self, background: pygame.Surface, foreground: pygame.Surface):
        self.background: pygame.Surface = background
        self.background_speed: float = 75
        self.foreground: pygame.Surface = foreground
        self.foreground_speed: float = 150

        self.start_x: int = (WIDTH - self.background.get_width()) // 2
        self.height: int = self.background.get_height()
        self.background_offset: int = HEIGHT - self.height
        self.foreground_offset: int = HEIGHT - self.height

    def update(self, dt: float):
        """Update game object in game loop

        Args:
            dt (float): time since last frame
        """

        self.background_offset += int(dt * self.background_speed)
        self.foreground_offset += int(dt * self.foreground_speed)

        if self.background_offset > HEIGHT:
            self.background_offset -= self.height
        if self.foreground_offset > HEIGHT:
            self.foreground_offset -= self.height

    def draw(self, window: pygame.Surface):
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        for index in range(2):
            window.blit(
                self.background,
                (self.start_x, self.background_offset - index * self.height),
            )
        for index in range(2):
            window.blit(
                self.foreground,
                (self.start_x, self.foreground_offset - index * self.height),
            )

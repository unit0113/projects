import pygame


class Timer:
    def __init__(self, duration) -> None:
        self.duration = duration
        self.active = False
        self.start_time = 0

    def activate(self):
        self.active = True
        self.start_time = pygame.time.get_ticks()

    def deactivate(self):
        self.active = False
        self.start_time = 0

    def update(self):
        if pygame.time.get_ticks() - self.start_time >= self.duration:
            self.deactivate()

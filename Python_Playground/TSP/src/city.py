import pygame
import numpy as np

from src.colors import LIGHT_GREY


class City:
    def __init__(self, window: pygame.surface.Surface, name: str, population: int, lat: float, long: float) -> None:
        self.name = name
        self.population = population
        self.lat = lat
        self.long = long
        self.x = None
        self.y = None
        self.window = window
    
    def distance_from(self, other: 'City') -> float:
        return np.sqrt((abs(self.lat - other.lat) ** 2) + (abs(self.long - other.long) ** 2))
    
    def __repr__(self):
        return self.name + ": (" + str(self.lat) + ", " + str(self.long) + ")"
    
    def draw(self):
        pygame.draw.circle(self.window, LIGHT_GREY, (self.x, self.y), 3)
import pygame
import numpy as np

from src.colors import MAXIMUM_GREEN
from src.settings import FL_N, FL_S, FL_E, FL_W

LARGE_CITY_SIZE = 4
STANDARD_CITY_SIZE = 3


class City:
    def __init__(self, window: pygame.surface.Surface, name: str, population: int, lat: str, long: str) -> None:
        self.name = name
        self.population = population
        self.lat = float(lat)
        self.long = float(long)
        self.x = None
        self.y = None
        self.window = window
        self.size = LARGE_CITY_SIZE if int(self.population) > 200_000 else STANDARD_CITY_SIZE
    
    def distance_from(self, other: 'City') -> float:
        """ Calculate the euclidian distance from this city to another city

        Args:
            other (City): Other city to caluclate the distance to

        Returns:
            float: Euclidian distance to the other city
        """

        return np.sqrt((abs(self.lat - other.lat) ** 2) + (abs(self.long - other.long) ** 2))
    
    def calculate_XY(self, image_start_x: int, image_start_y: int, image_height: int, image_width: int) -> None:
        """ Set the pixel X and Y values based on city lat/long and map image location and dimensions

        Args:
            image_start_x (int): X pixel location of top left corner of map
            image_start_y (int): Y pixel location of top left corner of map
            image_height (int): Height of map object
            image_width (int): Width of map object
        """

        self.x = (image_start_x + image_width)  - int((self.long  - FL_E) *  image_width / (FL_W - FL_E))
        self.y = (image_start_y + image_height) - int((self.lat - FL_S) * image_height / (FL_N - FL_S))
    
    def __repr__(self) -> str:
        """ Prints city information in the format: "Name: (lat, long)

        Returns:
            str: City information
        """
        return self.name + ": (" + str(self.lat) + ", " + str(self.long) + ")"
    
    def get_pixel_tuple(self) -> tuple[int, int]:
        """ Returns the X, Y values in the form of a tuple for drawing routes

        Returns:
            tuple[int, int]: X, Y values of the city
        """

        return (self.x, self.y)
    
    def draw(self):
        """ Draws a circle at the current X, Y location of the city
        """

        pygame.draw.circle(self.window, MAXIMUM_GREEN, (self.x, self.y), self.size)
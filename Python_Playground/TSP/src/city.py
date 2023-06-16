import numpy as np


class City:
    def __init__(self, window, lat, long):
        self.x = lat
        self.y = long
        self.window = window
    
    def distance_from(self, other):
        return np.sqrt((abs(self.x - other.x) ** 2) + (abs(self.y - other.y) ** 2))
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
    
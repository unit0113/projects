import random
import numpy as np

# City class with random initilization
class RandomCity:
    def __init__(self, map_size):
        self.x = int(random.random() * map_size)
        self.y = int(random.random() * map_size)
    
    def distance_from(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return id(self)
    
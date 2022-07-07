import numpy as np
import random

class Vector2D:
    def __init__(self, x=0, y=0):
        self._array = np.array([x, y], dtype=np.float32)

    @property
    def x(self):
        """The x-component of the vector"""
        return self._array[0]

    @x.setter
    def x(self, value):
        self._array[0] = value

    @property
    def y(self):
        """The y-component of the vector"""
        return self._array[1]

    @y.setter
    def y(self, value):
        self._array[1] = value

    @property
    def angle(self):
        return np.arctan2(self.y, self.x)

    @angle.setter
    def angle(self, theta):
        self.rotate(theta - self.angle)

    def get_angle(self):
        return np.arctan2(self.y, self.x)

    def normalize(self):
        self.magnitude = 1
        return self

    def distance(self, other):
        return np.sqrt(np.sum((self._array - other._array) ** 2))

    def __add__(self, other):
        x, y = self._array + other._array
        return self.__class__(x, y)

    def __sub__(self, other):
        x, y = self._array - other._array
        return self.__class__(x, y)

    def __mul__(self, k):
        """Multiply the point by a scalar"""
        if isinstance(k, int) or isinstance(k, float):
            x, y = k * self._array
            return self.__class__(x, y)
        raise TypeError("Can't multiply/divide a point by a non-numeric.")

    def rotate(self, theta):
        """Rotates the vector by an angle.
        :param theta: Angle (in radians).
        :type theta: float or int
        """
        x = self.x * np.cos(theta) - self.y * np.sin(theta)
        y = self.x * np.sin(theta) + self.y * np.cos(theta)
        self.x = x
        self.y = y

    @property
    def magnitude(self):
        """The magnitude of the vector.

        """
        return np.sqrt(np.dot(self._array, self._array))

    @magnitude.setter
    def magnitude(self, new_magnitude):
        current_magnitude = self.magnitude
        self._array = (new_magnitude * self._array) / current_magnitude

    def limit(self, upper_limit=None, lower_limit=None):
        magnitude = self.magnitude
        if upper_limit is None:
            upper_limit = magnitude
        if lower_limit is None:
            lower_limit = magnitude

        if magnitude < lower_limit:
            self.magnitude = lower_limit
        elif magnitude > upper_limit:
            self.magnitude = upper_limit

    def __gt__(self, scaler):
        return self.magnitude > scaler

    def __lt__(self, scaler):
        return self.magnitude < scaler

    @classmethod
    def random_2D(cls):
        """Return a random 2D unit vector.
        """
        x, y = 2 * (random(2) - 0.5)
        vec = cls(x, y)
        vec.normalize()
        return vec
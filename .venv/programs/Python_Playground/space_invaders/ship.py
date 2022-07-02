import random
from abc import ABC, abstractmethod
from lasers import Laser, LASER_SIZE


SHIP_SIZE = (50, 50)


class Ship(ABC):
    def __init__(self):
        self.ship_size = SHIP_SIZE
        self.laser_types = [self.laser1, self.laser2, self.laser3]
        self.laser_type_dmg_multipliers = [1, 0.65, 0.5]
        self.laser_type_current_index = 0

    @property
    def damage(self):
        return self.base_damage * random.uniform(0.5, 1.5) * self.laser_type_dmg_multipliers[self.laser_type_current_index]

    @property
    def is_dead(self):
        return self.health <= 0

    @property
    def center_points(self):
        x = self.rect.x + self.ship_size[0] // 2
        y = self.rect.y + self.ship_size[1] // 2
        return x, y

    def laser1(self):
        laser = Laser(self.rect.x + self.image.get_width() // 2 - LASER_SIZE[0] // 2, self.rect.y, self.damage)
        return [laser]

    def laser2(self):
        laser1 = Laser(self.rect.x, self.rect.y + 15, self.damage)
        laser2 = Laser(self.rect.x + SHIP_SIZE[0] - LASER_SIZE[0], self.rect.y + 15, self.damage)
        return [laser1, laser2]

    def laser3(self):
        laser1 = Laser(self.rect.x + self.image.get_width() // 2 - LASER_SIZE[0] // 2, self.rect.y, self.damage)
        laser2 = Laser(self.rect.x, self.rect.y + 15, self.damage)
        laser3 = Laser(self.rect.x + SHIP_SIZE[0] - LASER_SIZE[0], self.rect.y + 15, self.damage)
        
        return [laser1, laser2, laser3]

    def draw(self, window):
        window.blit(self.image, (self.rect.x, self.rect.y))

    @abstractmethod
    def fire(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def take_hit(self):
        pass

import pygame
import math
from settings import HEIGHT, FPS, MISSILE_SIZE, MISSILE_SPEED, MISSILE_BORESIGHT_VISION, MISSILE_VISION

class Missile:
    def __init__(self, x, y, damage, missile_image):
        self.damage = damage
        self.target = None
        self.lock = False
        self.image = missile_image
        self.image = pygame.transform.scale(self.image, MISSILE_SIZE)
        self.rect = pygame.Rect(x, y, self.image.get_width(), self.image.get_height())
        self.mask = pygame.mask.from_surface(self.image)
        self.timer = -1
        self.velocity = pygame.Vector2(0, 0)
        self.launch_profile = [4] * (FPS // 15) + [3] * (FPS // 15) + [2] * (FPS // 15) + [1] * (FPS // 15) + [0] * (FPS // 15) + [-1] * (FPS // 15) + [-2] * (FPS // 20) + [-3] * (FPS // 20) + [-4] * (FPS // 20) + [-5] * (FPS // 20) + [-6] * (FPS // 15) + [-7] *( FPS // 15) + [-8] * (FPS // 10)
        self.trail = []

    @property
    def is_off_screen(self):
        return self.rect.y > HEIGHT or self.rect.y < -MISSILE_SIZE[1]

    def check_possible_locks(self, baddies):
        lock = None
        distance = math.inf
        for baddie in baddies:
            if self.can_lock(baddie):
                new_distance = self.distance_to_object(baddie)
                if new_distance < distance:
                    lock = baddie
                    distance = new_distance
        
        if lock:
            self.target = lock

    def distance_to_object(self, other):
        return math.sqrt((self.rect.x - other.rect.x) ** 2 + (self.rect.y - other.rect.y) ** 2)

    def angle_between(self, other):
        x1, y1 = self.rect.x, self.rect.y
        x2, y2 = self.rect.x + self.velocity.x, self.rect.y + self.velocity.y
        x3, y3 = other.rect.x, other.rect.y
        deg1 = (360 + math.degrees(math.atan2(x1 - x2, y1 - y2))) % 360
        deg2 = (360 + math.degrees(math.atan2(x3 - x2, y3 - y2))) % 360
        return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

    def can_lock(self, object):
        return self.distance_to_object(object) < MISSILE_VISION and self.angle_between(object) < MISSILE_BORESIGHT_VISION

    def update(self):
        self.timer += 1
        if self.timer < len(self.launch_profile):
            self.rect.y += self.launch_profile[self.timer]
            return
        
        # else

    def draw(self, window):
        window.blit(self.image, (self.rect.x, self.rect.y))
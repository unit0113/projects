import pygame
import math
from settings import HEIGHT, WIDTH, FPS, MISSILE_SIZE, MISSILE_SPEED, MISSILE_BORESIGHT_VISION, MISSILE_VISION, MISSILE_TURN_RATE, PRO_NAV_CONST
import os

class Missile:
    def __init__(self, x, y, damage, missile_image):
        self.damage = damage
        self.target = None
        self.target_last_pos = None
        self.missile_last_pos = None
        self.image = missile_image
        self.image = pygame.transform.scale(self.image, MISSILE_SIZE)
        self.rect = pygame.Rect(x, y, self.image.get_width(), self.image.get_height())
        self.mask = pygame.mask.from_surface(self.image)
        self.timer = -1
        self.velocity = pygame.Vector2(0, -1)
        self.velocity.scale_to_length(MISSILE_SPEED // FPS)
        self.launch_profile = [speed for sublist in [[num] * (FPS // 15) for num in range(4, -1, -1)] for speed in sublist] + [speed for sublist in [[-num] * (FPS // 15) for num in range(1 + MISSILE_SPEED // FPS)] for speed in sublist]
        self.trail = []

    @property
    def is_off_screen(self):
        return (self.timer > len(self.launch_profile)
                and
                (self.rect.y > HEIGHT
                or self.rect.y < -MISSILE_SIZE[1]
                or self.rect.x - MISSILE_SIZE[1] > WIDTH
                or self.rect.x + MISSILE_SIZE[1] < 0))

    @property
    def can_check_locks(self):
        return self.timer > len(self.launch_profile) and not self.target

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
            self.target_last_pos = pygame.Vector2(*self.target.rect.center)
            self.missile_last_pos = pygame.Vector2(self.rect.x, self.rect.center[1])

    def distance_to_object(self, other):
        return math.sqrt((self.rect.x - other.rect.x) ** 2 + (self.rect.y - other.rect.y) ** 2)

    def angle_between(self, other):
        x1, y1 = self.velocity.x + self.rect.center[0], self.velocity.y + self.rect.center[1]
        x2, y2 = self.rect.center
        x3, y3 = other.rect.center
        v1 = (x1-x2, y1-y2)
        v2 = (x3-x2, y3-y2)

        return (math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])) * 360 / (2 * (math.pi))

    def can_lock(self, object):
        return self.distance_to_object(object) < MISSILE_VISION and abs(self.angle_between(object)) <= MISSILE_BORESIGHT_VISION

    def update(self):
        self.timer += 1
        if self.timer < len(self.launch_profile):
            self.rect.x += self.velocity.x
            self.rect.y += self.launch_profile[self.timer]
            return
        
        self.rect.x += self.velocity.x
        self.rect.y += self.velocity.y
        if self.target:
            self.pro_nav()
            if not self.can_lock(self.target):
                self.target = None
                self.target_last_pos = None
                self.missile_last_pos = None

    def pro_nav(self):
        # Source: https://www.moddb.com/members/blahdy/blogs/gamedev-introduction-to-proportional-navigation-part-i
        # Get missile to target vectors
        RTM_old = self.target_last_pos - self.missile_last_pos
        target_pos = pygame.Vector2(*self.target.rect.center) 
        missile_pos = pygame.Vector2(self.rect.x, self.rect.center[1])
        RTM_new = target_pos - missile_pos

        if not pygame.math.Vector2.length(RTM_old) or not pygame.math.Vector2.length(RTM_new):
            LOS_delta = pygame.Vector2(0, 0)
            LOS_rate = 0

        else:
            pygame.math.Vector2.normalize_ip(RTM_old)
            pygame.math.Vector2.normalize_ip(RTM_new)
            LOS_delta = RTM_new - RTM_old
            LOS_rate = pygame.math.Vector2.length(LOS_delta)

        # Range closing rate
        Vc = -LOS_rate

        # Final lateral acceleration
        latax = RTM_new * PRO_NAV_CONST * Vc * LOS_rate * LOS_delta

        # Prevent drift to left in to the left of target
        if self.rect.x < self.target.rect.x:
            latax *= -1

        # Update positions
        self.target_last_pos = target_pos
        self.missile_last_pos = missile_pos

        # Scale latax up to angle changes, clamp, and apply
        delta_angle = latax * 100_000_000     
        clamped_delta_angle = max(-MISSILE_TURN_RATE, min(MISSILE_TURN_RATE, PRO_NAV_CONST * delta_angle))
        self.velocity.rotate_ip(clamped_delta_angle)

    def draw(self, window):
        window.blit(pygame.transform.rotate(self.image, self.velocity.angle_to((0,-1))), (self.rect.x, self.rect.y))

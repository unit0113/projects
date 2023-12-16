import pygame
import math
from settings import (
    MISSILE_VISION,
    MISSILE_BORESIGHT,
    MISSILE_TURN_RATE,
    PRO_NAV_CONST,
)


class Missile:
    def __init__(
        self,
        pos: tuple[int, int],
        sprites: pygame.Surface,
        speed: int,
        direction: tuple[float, float],
        damage: float,
    ):
        self.damage = damage
        self.speed = speed
        self.target = None
        self.target_last_pos = None
        self.missile_last_pos = None

        self.sprites = sprites
        self.image = sprites[0]
        self.rect = self.image.get_rect()
        self.rect.midtop = pos
        self.mask = pygame.mask.from_surface(self.image)
        self.timer = 0
        self.direction = direction
        self.vel_vector = pygame.Vector2(0, -direction[1])
        self.accel_vector = pygame.Vector2(0, 0)
        self.state = 0

        self.trail = []

    @property
    def is_seeker_active(self) -> bool:
        return self.state == 0 and not self.target

    def check_possible_locks(self, enemies: pygame.sprite.Group) -> None:
        """Find the closest lockable enemy and lock onto them

        Args:
            enemies (pygame.sprite.Group): sprite group of enemies
        """
        lock = None
        min_distance = math.inf
        # Find closest lockable enemy
        for enemy in enemies:
            if self.can_lock(enemy):
                new_distance = self._distance_to_object(enemy)
                if new_distance < min_distance:
                    lock = enemy
                    min_distance = new_distance

        if lock:
            self.target = lock
            self.target_last_pos = pygame.Vector2(*self.target.rect.center)
            self.missile_last_pos = pygame.Vector2(*self.rect.center)

    def _distance_to_object(self, other: object) -> float:
        """Finds the distance from the missile to the potential target

        Args:
            other (object): Possible enemy

        Returns:
            float: distance to the object
        """
        return math.sqrt(
            (self.rect.x - other.rect.x) ** 2 + (self.rect.y - other.rect.y) ** 2
        )

    def angle_between(self, other: object) -> float:
        """Calculates the angle off boresite of the potential target

        Args:
            other (object): Possible enemy

        Returns:
            float: angle to the object
        """
        x1, y1 = (
            self.vel_vector.x + self.rect.center[0],
            self.vel_vector.y + self.rect.center[1],
        )
        x2, y2 = self.rect.center
        x3, y3 = other.rect.center
        v1 = (x1 - x2, y1 - y2)
        v2 = (x3 - x2, y3 - y2)

        return (
            (math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
            * 360
            / (2 * (math.pi))
        )

    def can_lock(self, object):
        return (
            self._distance_to_object(object) < MISSILE_VISION
            and abs(self.angle_between(object)) <= MISSILE_BORESIGHT
        )

    def update(self, dt: float):
        # If in launch profile
        self.timer += dt
        if self.state == 0 and self.timer > 2:
            self.accel_vector.y = 0.02
        if self.timer < len(self.launch_profile):
            self.rect.x += self.vel_vector.x
            self.rect.y += self.launch_profile[self.timer]
            return

        self.rect.x += self.vel_vector.x
        self.rect.y += self.vel_vector.y

        # If guiding to target
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

        if not pygame.math.Vector2.length(RTM_old) or not pygame.math.Vector2.length(
            RTM_new
        ):
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
        clamped_delta_angle = max(
            -MISSILE_TURN_RATE, min(MISSILE_TURN_RATE, PRO_NAV_CONST * delta_angle)
        )
        self.vel_vector.rotate_ip(clamped_delta_angle)

    def draw(self, window):
        window.blit(
            pygame.transform.rotate(self.image, self.vel_vector.angle_to((0, -1))),
            (self.rect.x, self.rect.y),
        )

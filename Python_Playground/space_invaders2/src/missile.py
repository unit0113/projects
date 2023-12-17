import pygame
import math
from .settings import (
    MISSILE_VISION,
    MISSILE_BORESIGHT,
    MISSILE_TURN_RATE,
    PRO_NAV_CONST,
    FRAME_TIME,
)


class Missile(pygame.sprite.Sprite):
    def __init__(
        self,
        sprites: pygame.Surface,
        pos: tuple[int, int],
        damages: tuple[float, float],
        speed: int,
        direction: tuple[float, float],
    ):
        pygame.sprite.Sprite.__init__(self)
        self.shield_damage, self.ship_damage = damages
        self.speed = speed
        self.target = None
        self.target_last_pos = None
        self.missile_last_pos = None

        self.frame_index = 0
        self.animation_frames = [0, 0, 1]
        self.last_frame = pygame.time.get_ticks()
        self.sprites = sprites
        self.original_image = sprites[0]
        self.image = self.original_image
        self.rect = self.original_image.get_rect()
        self.pos = pos
        self.rect.midtop = pos
        self.mask = pygame.mask.from_surface(self.original_image)

        self.timer = 0
        self.direction = direction
        self.vel_vector = pygame.Vector2(0, 0.25 * -direction[1])
        self.accel_vector = pygame.Vector2(0, 0)
        self.state = 0

    @property
    def is_seeker_active(self) -> bool:
        return self.state == 2

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

    def update(self, dt: float, enemies: pygame.sprite.Group):
        """Update game object in game loop

        Args:
            dt (float): time since last frame
            enemies (pygame.sprite.Group): enemies currently in game
        """

        # Launch profile state machine
        self.timer += dt
        if self.state == 0 and self.timer > 0.5:
            self.accel_vector.y = 0.02 * self.direction[1]
            self.state = 1
        elif self.state == 1 and self.vel_vector.magnitude() > 1:
            self.accel_vector.y = 0
            self.state = 2
            self.animation_frames = [2, 2, 1]

        self.animate()

        # If guiding to target
        if self.is_seeker_active:
            if not self.target:
                self.check_possible_locks(enemies)
            if self.target:
                self.pro_nav()
                if not self.can_lock(self.target):
                    self.target = None
                    self.target_last_pos = None
                    self.missile_last_pos = None

            # Rotate image based on velocity
            self.image = pygame.transform.rotate(
                self.original_image, self.vel_vector.angle_to(self.direction)
            )

        # Update position and velocity
        self.pos += (
            self.vel_vector * self.speed * dt
            + 0.5 * self.accel_vector * dt * dt * self.speed
        )
        self.rect.center = self.pos
        self.vel_vector += self.accel_vector

    def animate(self) -> None:
        """Controls sprite animation of missile"""

        if pygame.time.get_ticks() > self.last_frame + FRAME_TIME:
            # Reset frame counter and increment frame
            self.last_frame = pygame.time.get_ticks()
            self.frame_index += 1
            if self.frame_index >= len(self.animation_frames):
                # Loop animation
                self.frame_index = 0
            self.original_image = self.sprites[self.animation_frames[self.frame_index]]

    def pro_nav(self):
        """Implement proportial navigations as laid out in
        https://www.moddb.com/members/blahdy/blogs/gamedev-introduction-to-proportional-navigation-part-i
        """

        # Get missile to target vectors
        RTM_old = self.target_last_pos - self.missile_last_pos
        target_pos = pygame.Vector2(self.target.rect.center)
        missile_pos = pygame.Vector2(self.rect.center)
        RTM_new = target_pos - missile_pos

        # Update positions
        self.target_last_pos = target_pos
        self.missile_last_pos = missile_pos

        if not pygame.math.Vector2.length(RTM_old) or not pygame.math.Vector2.length(
            RTM_new
        ):
            return

        RTM_old.normalize_ip()
        RTM_new.normalize_ip()
        LOS_delta = RTM_new - RTM_old
        LOS_rate = LOS_delta.length()

        # Closing velocity
        Vc = -LOS_rate

        # Final lateral acceleration
        latax = RTM_new * PRO_NAV_CONST * PRO_NAV_CONST * Vc * LOS_rate * LOS_delta / 2

        # Prevent drift to left if to the left of target
        # Occurs during near straight on engagements
        if latax < 0 and RTM_new.x - RTM_old.x > 0:
            latax *= -1

        # Scale latax up, clamp, and apply
        clamped_delta_angle = max(
            -MISSILE_TURN_RATE,
            min(MISSILE_TURN_RATE, latax * 1_000_000),
        )
        self.vel_vector.rotate_ip(clamped_delta_angle)

    def get_shield_damage(self) -> float:
        """Getter for shield damage

        Returns:
            float: damage done to shields
        """
        return self.shield_damage

    def get_ship_damage(self) -> float:
        """Getter for ship damage

        Returns:
            float: damage done to ships
        """
        return self.ship_damage

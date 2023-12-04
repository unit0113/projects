import pygame
from pygame.math import Vector2
import math


class Enemy(pygame.sprite.Sprite):
    def __init__(
        self,
        waypoints: list[list[int, int]],
        image: pygame.surface.Surface,
        health: int,
        move_speed: float,
    ) -> None:
        pygame.sprite.Sprite.__init__(self)
        self.waypoints = waypoints
        self.pos = Vector2(self.waypoints[0])
        self.target_waypoint_index = 1
        self.original_image = image

        self.angle = 0
        self.move_speed = move_speed
        self.health = health
        self.reward = self.move_speed * self.health

        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos

    def update(self, world) -> None:
        self._move(world)
        self._rotate()

    def _move(self, world) -> None:
        if self.target_waypoint_index < len(self.waypoints):
            self.movement = (
                Vector2(self.waypoints[self.target_waypoint_index]) - self.pos
            )
            dist = self.movement.length()
            if dist > self.move_speed:
                self.pos += self.movement.normalize()
            else:
                if dist != 0:
                    self.pos += self.movement.normalize() * dist
                self.target_waypoint_index += 1
        else:
            self.kill()
            world.take_damage(1)

    def _rotate(self) -> None:
        self.angle = math.degrees(math.atan2(-self.movement[1], self.movement[0]))
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos

    def take_damage(self, dmg: int) -> int:
        self.health -= dmg
        if self.health <= 0:
            self.kill()
            return self.reward
        return 0

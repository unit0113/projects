import pygame
import random
import math

from .entities import PhysicsEntity
from .spark import Spark
from .particle import Particle


class Enemy(PhysicsEntity):
    def __init__(self, game, pos, size) -> None:
        super().__init__(game, "enemy", pos, size)

        self.walking = 0

    def update(self, tile_map, movement=(0, 0)):
        if self.walking:
            if tile_map.solid_tile_check(
                (self.rect().centerx + (-7 if self.flip else 7), self.pos[1] + 23)
            ):
                if self.collisions["right"] or self.collisions["left"]:
                    self.flip = not self.flip
                movement = (-0.5 if self.flip else 0.5, movement[1])
            else:
                self.flip = not self.flip
            self.walking = max(0, self.walking - 1)
            if not self.walking:
                dis = (
                    self.game.player.pos[0] - self.pos[0],
                    self.game.player.pos[1] - self.pos[1],
                )
                if abs(dis[1]) < 16:
                    self.game.sfx["shoot"].play()
                    if self.flip and dis[0] < 0:
                        self.game.projectiles.append(
                            [[self.rect().centerx - 7, self.rect().centery], -1.5, 0]
                        )
                        for _ in range(4):
                            self.game.sparks.append(
                                Spark(
                                    self.game.projectiles[-1][0],
                                    random.random() - 0.5 + math.pi,
                                    2 + random.random(),
                                )
                            )
                    elif not self.flip and dis[0] > 0:
                        self.game.projectiles.append(
                            [[self.rect().centerx + 7, self.rect().centery], 1.5, 0]
                        )
                        for _ in range(4):
                            self.game.sparks.append(
                                Spark(
                                    self.game.projectiles[-1][0],
                                    random.random() - 0.5,
                                    2 + random.random(),
                                )
                            )
        elif random.random() < 0.01:
            self.walking = random.randint(30, 120)

        super().update(tile_map, movement)

        if movement[0]:
            self.set_action("run")
        else:
            self.set_action("idle")

        if abs(self.game.player.dashing) >= 50:
            if self.rect().colliderect(self.game.player.rect()):
                self.game.sfx["hit"].play()
                self.game.screen_shake = max(16, self.game.screen_shake)
                for _ in range(30):
                    angle = random.random() * math.pi * 2
                    speed = 5 * random.random()
                    self.game.sparks.append(
                        Spark(
                            self.rect().center,
                            angle,
                            2 + random.random(),
                        )
                    )
                    self.game.particles.append(
                        Particle(
                            self.game,
                            "particle",
                            self.rect().center,
                            [
                                math.cos(angle + math.pi) * speed * 0.5,
                                math.sin(angle + math.pi) * speed * 0.5,
                            ],
                            frame=random.randint(0, 7),
                        )
                    )
                self.game.sparks.append(
                    Spark(self.rect().center, 0, 5 + random.random())
                )
                self.game.sparks.append(
                    Spark(self.rect().center, math.pi, 5 + random.random())
                )
                return True
        return False

    def draw(self, window, offset=(0, 0)):
        super().draw(window, offset)
        if self.flip:
            window.blit(
                pygame.transform.flip(self.game.assets["gun"], True, False),
                (
                    self.rect().centerx
                    - 4
                    - self.game.assets["gun"].get_width()
                    - offset[0],
                    self.rect().centery - offset[1],
                ),
            )
        else:
            window.blit(
                self.game.assets["gun"],
                (self.rect().centerx + 4 - offset[0], self.rect().centery - offset[1]),
            )

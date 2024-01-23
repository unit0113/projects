import math
import random

from .entities import PhysicsEntity
from .particle import Particle
from .settings import TERMINAL_VELOCITY


class Player(PhysicsEntity):
    def __init__(self, game, pos, size) -> None:
        super().__init__(game, "player", pos, size)
        self.air_time = 0
        self.available_jumps = 1
        self.wall_slide = False
        self.dashing = 0

    def update(self, tilemap, movement=(0, 0)):
        super().update(tilemap, movement)

        self.air_time += 1
        if self.air_time > 120 and self.velocity[1] == TERMINAL_VELOCITY:
            self.game.dead += 1
            self.game.screen_shake = max(16, self.game.screen_shake)
            return

        if self.collisions["down"]:
            self.air_time = 0
            self.available_jumps = 1

        self.wall_slide = False
        if (self.collisions["right"] or self.collisions["left"]) and self.air_time > 4:
            self.wall_slide = True
            self.velocity[1] = min(self.velocity[1], 0.5)
            if self.collisions["right"]:
                self.flip = False
            else:
                self.flip = True
            self.set_action("wall_slide")

        else:
            if self.air_time > 4:
                self.set_action("jump")
            elif movement[0] != 0:
                self.set_action("run")
            else:
                self.set_action("idle")

        if self.dashing > 0:
            self.dashing = max(0, self.dashing - 1)
        elif self.dashing < 0:
            self.dashing = min(0, self.dashing + 1)

        if abs(self.dashing) > 50:
            self.velocity[0] = abs(self.dashing) / self.dashing * 8
            if abs(self.dashing) == 51:
                self.velocity[0] *= 0.1
            self.game.particles.append(
                Particle(
                    self.game,
                    "particle",
                    self.rect().center,
                    [abs(self.dashing) / self.dashing * random.random() * 3, 0],
                    frame=random.randint(0, 7),
                )
            )

        # Burst at start and end of dash
        if abs(self.dashing) in {60, 50}:
            for _ in range(20):
                angle = random.random() * math.pi * 2
                speed = random.random() * 0.5 + 0.5
                self.game.particles.append(
                    Particle(
                        self.game,
                        "particle",
                        self.rect().center,
                        [math.cos(angle) * speed, math.sin(angle) * speed],
                        frame=random.randint(0, 7),
                    )
                )

        # Normalize x vel (air resistance)
        if self.velocity[0] > 0:
            self.velocity[0] = max(self.velocity[0] - 0.1, 0)
        if self.velocity[0] < 0:
            self.velocity[0] = min(self.velocity[0] + 0.1, 0)

    def jump(self):
        if self.wall_slide:
            if self.flip and self.last_movemement[0] < 0:
                self.velocity[0] = 3.5
                self.velocity[1] = -2.5
                self.air_time = 5
                self.available_jumps = max(0, self.available_jumps - 1)
                return True
            elif not self.flip and self.last_movemement[0] > 0:
                self.velocity[0] = -3.5
                self.velocity[1] = -2.5
                self.air_time = 5
                self.available_jumps = max(0, self.available_jumps - 1)
                return True
        elif self.available_jumps:
            self.velocity[1] = -3
            self.available_jumps -= 1
            self.air_time = 5
            return True

    def dash(self):
        if not self.dashing:
            self.game.sfx["dash"].play()
            self.dashing = -61 if self.flip else 61

    def draw(self, window, offset=(0, 0)):
        if abs(self.dashing) <= 50:
            super().draw(window, offset=offset)
        else:
            pass

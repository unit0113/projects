import pygame

from .settings import TERMINAL_VELOCITY, GRAVITY


class PhysicsEntity:
    def __init__(self, game, e_type, pos, size) -> None:
        self.game = game
        self.e_type = e_type

        # Prevent pass by reference when passed to multiple entities
        self.pos = list(pos)
        self.size = size
        self.velocity = [0, 0]
        self.collisions = {"up": False, "down": False, "left": False, "right": False}

        # Animation
        self.action = ""
        self.anim_offset = (-3, -3)  # Accounts for different sprite sizes
        self.flip = False
        self.set_action("idle")

        self.last_movemement = [0, 0]

    def set_action(self, action):
        if self.action != action:
            self.action = action
            self.animation = self.game.assets[f"{self.e_type}/{self.action}"].copy()

    def rect(self):
        return pygame.Rect(self.pos[0], self.pos[1], self.size[0], self.size[1])

    def update(self, tilemap, movement=(0, 0)) -> None:
        self.collisions = {"up": False, "down": False, "left": False, "right": False}

        frame_movement = (
            movement[0] + self.velocity[0],
            movement[1] + self.velocity[1],
        )

        self.pos[0] += frame_movement[0]
        entity_rect = self.rect()
        for rect in self.game.tile_map.physics_rects_around(self.pos):
            if entity_rect.colliderect(rect):
                if frame_movement[0] > 0:
                    entity_rect.right = rect.left
                    self.collisions["right"] = True
                elif frame_movement[0] < 0:
                    entity_rect.left = rect.right
                    self.collisions["left"] = True
                self.pos[0] = entity_rect.x

        self.pos[1] += frame_movement[1]
        entity_rect = self.rect()
        for rect in self.game.tile_map.physics_rects_around(self.pos):
            if entity_rect.colliderect(rect):
                if frame_movement[1] > 0:
                    entity_rect.bottom = rect.top
                    self.collisions["down"] = True
                    self.velocity[1] = 0
                elif frame_movement[1] < 0:
                    entity_rect.top = rect.bottom
                    self.collisions["up"] = True
                self.pos[1] = entity_rect.y

        if movement[0] > 0:
            self.flip = False
        elif movement[0] < 0:
            self.flip = True

        self.last_movemement = movement

        self.velocity[1] = min(TERMINAL_VELOCITY, self.velocity[1] + GRAVITY)

        if self.collisions["up"] or self.collisions["down"]:
            self.velocity[1] = 0

        # Update animation
        self.animation.update()

    def draw(self, window: pygame.Surface, offset) -> None:
        window.blit(
            pygame.transform.flip(self.animation.img(), self.flip, False),
            (
                self.pos[0] - offset[0] + self.anim_offset[0],
                self.pos[1] - offset[1] + self.anim_offset[1],
            ),
        )

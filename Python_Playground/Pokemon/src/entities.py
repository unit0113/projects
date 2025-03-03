from settings import *


class Entity(pygame.sprite.Sprite):
    def __init__(self, pos, frames, groups, facing_direction):
        super().__init__(groups)

        # Graphics
        self.frame_index = 0
        self.frames = frames
        self.facing_direction = facing_direction

        # Movement
        self.direction = vector()
        self.speed = 250

        # Sprites
        self.image = self.frames[self.get_state()][self.frame_index]
        self.rect = self.image.get_frect(center=pos)

    def animate(self, dt):
        self.frame_index += ANIMATION_SPEED * dt
        self.image = self.frames[self.get_state()][
            int(self.frame_index) % len(self.frames[self.get_state()])
        ]

    def get_state(self):
        moving = bool(self.direction)
        if moving:
            if self.direction.x != 0:
                self.facing_direction = "right" if self.direction.x > 0 else "left"
            if self.direction.y != 0:
                self.facing_direction = "down" if self.direction.y > 0 else "up"
        return f"{self.facing_direction}{'' if moving else '_idle'}"


class Character(Entity):
    def __init__(self, pos, frames, groups, facing_direction):
        super().__init__(pos, frames, groups, facing_direction)


class Player(Entity):
    def __init__(self, pos, frames, groups, facing_direction):
        super().__init__(pos, frames, groups, facing_direction)
        self.direction = vector()

    def input(self):
        keys = pygame.key.get_pressed()
        input_vector = vector()
        if keys[pygame.K_UP]:
            input_vector.y -= 1
        if keys[pygame.K_DOWN]:
            input_vector.y += 1
        if keys[pygame.K_LEFT]:
            input_vector.x -= 1
        if keys[pygame.K_RIGHT]:
            input_vector.x += 1

        self.direction = input_vector

    def move(self, dt):
        self.rect.center += self.direction * self.speed * dt

    def update(self, dt):
        self.input()
        self.move(dt)
        self.animate(dt)

import pygame
from math import cos, sin, pi

CART_COLOR = (7, 132, 181)
WHEEL_COLOR = (57, 172, 231)
POLE_COLOR = (136, 153, 166)
POLE_END_COLOR = (165, 0, 0)
CART_SPEED = 900
FPS = 60
GRAVITY = 9.8
FRICTION = 0.02
DAMPING_FACTOR = 0.1

class Cart:
    def __init__(self, window, base_height):
        self.window = window

        self.window_width, self.window_height = self.window.get_size()
        body_width = self.window_width // 10
        body_height = self.window_height // 20
        self.wheel_radius = self.window_height // 40
        self.body = pygame.Rect(self.window_width // 2 - body_width // 2, base_height - body_height - self.wheel_radius, body_width, body_height)
        
        self.pole_angle = pi
        self.angular_velocity = 0
        self.pole_length = body_width

    def draw(self):
        self.update()

        pygame.draw.rect(self.window, CART_COLOR, self.body)
        pygame.draw.circle(self.window, WHEEL_COLOR, (self.body.x + 1.5 * self.wheel_radius, self.body.y + self.body.height), self.wheel_radius)
        pygame.draw.circle(self.window, WHEEL_COLOR, (self.body.x + self.body.width - 1.5 * self.wheel_radius, self.body.y + self.body.height), self.wheel_radius)
        pygame.draw.line(self.window, POLE_COLOR, (self.body.center), (self.body.centerx + cos(self.pole_angle) * self.pole_length, self.body.centery + sin(self.pole_angle) * self.pole_length), 4)
        pygame.draw.circle(self.window, POLE_END_COLOR, (self.body.centerx + cos(self.pole_angle) * self.pole_length, self.body.centery + sin(self.pole_angle) * self.pole_length), 5)

    def update(self):
        self.angular_velocity += DAMPING_FACTOR * cos(self.pole_angle) * self.pole_length * GRAVITY / (2 * FPS)
        self.angular_velocity *= 1 - FRICTION
        self.pole_angle += self.angular_velocity / FPS

    def left(self):
        if self.body.x > 0 + CART_SPEED // FPS:
            self.body.x -= CART_SPEED // FPS
        else:
            self.body.x = 0

    def right(self):
        if self.body.x + self.body.width < self.window_width - CART_SPEED // FPS:
            self.body.x += CART_SPEED // FPS
        else:
            self.body.x = self.window_width - self.body.width
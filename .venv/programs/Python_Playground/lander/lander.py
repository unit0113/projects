import pygame
from screeninfo import get_monitors
from vector import Vector2D
import math


# Constants
monitor = get_monitors()[0]
WIDTH = monitor.width
HEIGHT = monitor.height
DEFAULT_IMAGE_SIZE = (30, 30)
RCS_STRENGTH = 5
RCS_FUEL_BURN = 3
MAIN_THRUSTER_STRENGTH = 5
MAIN_THRUSTER_FUEL_BURN = 10
LANDER_MASS = 100
GRAVITY = 1.62
FUEL_AMOUNT = 5000
FPS = 100

# Colors
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (220, 220, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
L_GRAY = (200, 200, 200)
D_GRAY = (70, 70, 70)

class Lander:
    def __init__(self, window):
        self.window = window
        image = pygame.image.load(r'Python_Playground\lander\Lunar Lander.png')
        self.image = pygame.transform.scale(image, DEFAULT_IMAGE_SIZE)
        self.rect = self.image.get_rect()
        self.rect.x = WIDTH // 2 - self.image.get_width() // 2
        self.rect.y = HEIGHT // 2 - self.image.get_height()
        self.velocity = Vector2D(0,0)
        self.orientation = Vector2D(0, 1)
        self.initialize_fuel()

    def initialize_fuel(self):
        self.fuel_gauge_border = 2
        self.fuel_gauge_height = 400
        self.fuel_gauge_width = 25
        self.fuel = self.max_fuel = FUEL_AMOUNT
        self.fuel_gauge_background = pygame.Rect(WIDTH - 75, 30, self.fuel_gauge_width, self.fuel_gauge_height)
        self.fuel_gauge_middle = pygame.Rect(WIDTH - 75 + self.fuel_gauge_border, 30 + self.fuel_gauge_border, self.fuel_gauge_width - 2 * self.fuel_gauge_border, self.fuel_gauge_height - 2 * self.fuel_gauge_border)

    @property
    def fuel_percentage(self):
        return self.fuel / self.max_fuel

    @property
    def fuel_gauge_fuel_level(self):
        return pygame.Rect(WIDTH - 75 + self.fuel_gauge_border, self.fuel_gauge_height * (1 - self.fuel_percentage) + (30 + self.fuel_gauge_border), self.fuel_gauge_width - 2 * self.fuel_gauge_border, (self.fuel_percentage) * (self.fuel_gauge_height - 2 * self.fuel_gauge_border))

    def draw(self):
        self.draw_fuel_gauge()
        self.velocity.y += GRAVITY / FPS
        self.rect.x += self.velocity.x
        self.rect.y += self.velocity.y
        image = pygame.transform.rotate(self.image, 90 - self.orientation.angle * 360 / (2  * math.pi))
        self.window.blit(image, (self.rect.x, self.rect.y))

    def draw_fuel_gauge(self):
        pygame.draw.rect(self.window, WHITE, self.fuel_gauge_background)
        pygame.draw.rect(self.window, BLACK, self.fuel_gauge_middle)
        if self.fuel_percentage > 0.66:
            color = GREEN
        elif self.fuel_percentage <= 0.33:
            color = RED
        else:
            color = YELLOW
        pygame.draw.rect(self.window, color, self.fuel_gauge_fuel_level)
    
    def main_thuster(self):
        if self.fuel <= 0:
            return
        self.velocity -= self.orientation * (MAIN_THRUSTER_STRENGTH / FPS)
        self.fuel -= MAIN_THRUSTER_FUEL_BURN

    def right_thruster(self):
        if self.fuel <= 0:
            return
        self.orientation.rotate(-RCS_STRENGTH / FPS)
        self.fuel -= RCS_FUEL_BURN

    def left_thruster(self):
        if self.fuel <= 0:
            return
        self.orientation.rotate(RCS_STRENGTH / FPS)
        self.fuel -= RCS_FUEL_BURN


class Terrain:
    def __init__(self, window):
        self.window = window

    def draw(self):
        self.window.fill(BLACK)


def draw(terrain, lander):
    terrain.draw()
    lander.draw()
    pygame.display.update()


def initialize_pygame():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ants")
    terrain = Terrain(window)
    lander = Lander(window)

    return window, terrain, lander


def main():
    window, terrain, lander = initialize_pygame()
    clock = pygame.time.Clock()
    draw(terrain, lander)

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_q]:
            pygame.quit()
            quit()

        if keys[pygame.K_r]:
            main()

        if keys[pygame.K_UP]:
            lander.main_thuster()

        if keys[pygame.K_LEFT]:
            lander.right_thruster()
        
        if keys[pygame.K_RIGHT]:
            lander.left_thruster()

        draw(terrain, lander)


if __name__ == "__main__":
    main()
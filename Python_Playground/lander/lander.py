import pygame
from screeninfo import get_monitors
from vector import Vector2D
import math
import random


# Constants
monitor = get_monitors()[0]
WIDTH = monitor.width
HEIGHT = monitor.height
DEFAULT_IMAGE_SIZE = (30, 30)
RCS_STRENGTH = 150
RCS_FUEL_BURN = 3
MAIN_THRUSTER_STRENGTH = 25000
MAIN_THRUSTER_FUEL_BURN = MAIN_THRUSTER_STRENGTH / 2000
LANDER_MASS = 2000
PER_UNIT_FUEL_MASS = 2
GRAVITY = 1.62
FUEL_AMOUNT = 5000
DEATH_SINK_RATE = 2.5
DEATH_LATERAL_VELOCITY = 0.25
LANDING_AREA_SIZE = 200
TERRAIN_BUMPYNESS = 125
FPS = 30

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
        self.rotational_velocity = 0
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

    @property
    def mass(self):
        return LANDER_MASS + self.fuel * PER_UNIT_FUEL_MASS

    def draw(self):
        self.draw_gui()
        self.velocity.y += GRAVITY / FPS
        self.rect.x += self.velocity.x
        self.rect.y += self.velocity.y
        self.orientation.rotate(self.rotational_velocity / FPS)
        image = pygame.transform.rotate(self.image, 90 - self.orientation.angle * 360 / (2  * math.pi))
        self.window.blit(image, (self.rect.x, self.rect.y))

    def draw_gui(self):
        self.draw_fuel_gauge()
        self.draw_vert_vel()
        self.draw_lateral_vel()

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

    def draw_vert_vel(self):
        if abs(self.velocity.y) > DEATH_SINK_RATE:
            color = RED
        elif abs(self.velocity.y) < DEATH_SINK_RATE / 2:
            color = GREEN
        else:
            color = YELLOW
            
        velocity_text = FONT.render(f'Sink Rate: {FPS * self.velocity.y:.2f}', 1, color)
        self.window.blit(velocity_text, (WIDTH - 150 - velocity_text.get_width(), 40 - velocity_text.get_height() // 2))

    def draw_lateral_vel(self):
        if abs(self.velocity.x) > DEATH_LATERAL_VELOCITY:
            color = RED
        elif abs(self.velocity.x) < DEATH_LATERAL_VELOCITY / 2:
            color = GREEN
        else:
            color = YELLOW
            
        velocity_text = FONT.render(f'Lateral Velocity: {FPS * self.velocity.x:.2f}', 1, color)
        self.window.blit(velocity_text, (WIDTH - 150 - velocity_text.get_width(), 70 - velocity_text.get_height() // 2))
    
    def main_thuster(self):
        if self.fuel <= 0:
            return
        self.velocity -= self.orientation * ((MAIN_THRUSTER_STRENGTH / FPS)/ self.mass)
        self.fuel -= MAIN_THRUSTER_FUEL_BURN

    def right_thruster(self):
        if self.fuel <= 0:
            return
        self.rotational_velocity -= RCS_STRENGTH / self.mass
        self.fuel -= RCS_FUEL_BURN

    def left_thruster(self):
        if self.fuel <= 0:
            return
        self.rotational_velocity += RCS_STRENGTH / self.mass
        self.fuel -= RCS_FUEL_BURN


class Terrain:
    def __init__(self, window):
        self.window = window
        landing_area_left = (random.randint(LANDING_AREA_SIZE, WIDTH - 2 * LANDING_AREA_SIZE), self.get_rand_height())
        landing_area_right = (landing_area_left[0] + LANDING_AREA_SIZE, landing_area_left[1])

        self.poly_points = []
        left_num_points = random.randint(int(0.75 * landing_area_left[0] / TERRAIN_BUMPYNESS), int(1.25 * landing_area_left[0] // TERRAIN_BUMPYNESS))
        left_points = random.sample(range(landing_area_left[0]), left_num_points)
        for point in sorted(left_points):
            self.poly_points.append((point, self.get_rand_height()))

        self.poly_points.append(landing_area_left)
        self.poly_points.append(landing_area_right)

        right_num_points = random.randint(int(0.75 * (WIDTH - landing_area_right[0]) / TERRAIN_BUMPYNESS), int(1.25 * (WIDTH - landing_area_right[0]) // TERRAIN_BUMPYNESS))
        right_points = random.sample(range(landing_area_right[0], WIDTH+1), right_num_points)
        for point in sorted(right_points):
            self.poly_points.append((point, self.get_rand_height()))

        self.poly_points += [(WIDTH, HEIGHT - 200), (WIDTH, HEIGHT), (0, HEIGHT), (0, HEIGHT - 200)]


    def get_rand_height(self):
        return HEIGHT - random.randint(200, 600)

    def draw(self):
        self.window.fill(BLACK)
        pygame.draw.polygon(self.window, WHITE, self.poly_points)


def draw(terrain, lander):
    terrain.draw()
    lander.draw()
    pygame.display.update()


def initialize_pygame():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Lander")
    global FONT
    FONT = pygame.font.SysFont('verdana', 20, bold=True)
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
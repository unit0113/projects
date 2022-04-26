import pygame
pygame.init()
import math
import random

WIDTH, HEIGHT = 1400, 1400
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Planet Simulation")

# Colors
RED = (255, 0, 0)
LTRED = (188, 39, 50)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
LTBLUE = (100, 149, 245)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
DARK_GRAY = (80, 78, 81)

FONT = pygame.font.SysFont("comicsans", 16)


def random_start_location(distance):
    M1 = 1.98892e30

    angle = 2 * math.pi * random.random()
    x = distance * math.cos(angle)
    y = distance * math.sin(angle)

    if distance != 0:
        velocity = math.sqrt(Planet.G * M1 / distance)
        vy = abs(velocity * math.cos(angle))
        if math.pi / 2 <= angle < 1.5 * math.pi:
            vy = vy * -1

        vx = abs(velocity * math.sin(angle))
        if 0 <= angle < math.pi:
            vx = vx * -1
        
    else:
        vx = vy = 0

    return x, y, vx, vy


class Planet:
    AU = 149.6e9
    G = 6.67428e-11
    SCALE = 125 / AU # 1AU = 100 pixels
    TIMESTEP = 3600 * 24 # 1 day per tick

    def __init__(self, distance_to_sun, radius, color, mass, draw_orbit=True):
        self.x, self.y, self.x_vel, self.y_vel = random_start_location(distance_to_sun * Planet.AU)
        self.distance_to_sun = distance_to_sun
        self.radius = radius
        self.color = color
        self.mass = mass
        self.orbit = []


    def draw(self, window):
        x = self.x * self.SCALE + WIDTH / 2
        y = self.y * self.SCALE + HEIGHT / 2

        # Reorg with if draw_orbit

        if len(self.orbit) > 2:
            updated_points = []
            for point in self.orbit:
                x, y = point
                x = x * self.SCALE + WIDTH / 2
                y = y * self.SCALE + HEIGHT / 2
                updated_points.append((x, y))

            pygame.draw.lines(window, self.color, False, updated_points, 2)

        pygame.draw.circle(window, self.color, (x, y), self.radius)
		
        if self.mass != 1.98892e30:
            distance_text = FONT.render(f"{self.distance_to_sun:.4f}AU", 1, WHITE)
            window.blit(distance_text, (x - distance_text.get_width()/2, y - distance_text.get_height()/2))


    def attraction(self, other):
        distance_x = other.x - self.x
        distance_y = other.y - self.y
        distance = math.sqrt(distance_x**2 + distance_y**2)
        self.distance_to_sun = distance / Planet.AU

        force = self.G * self.mass * other.mass / distance ** 2
        theta = math.atan2(distance_y, distance_x)
        force_x = force * math.cos(theta)
        force_y = force * math.sin(theta)

        return force_x, force_y


    def update_position(self, sun):
        fx, fy = self.attraction(sun)

        self.x_vel += (fx / self.mass) * self.TIMESTEP
        self.y_vel += (fy / self.mass) * self.TIMESTEP

        self.x += self.x_vel * self.TIMESTEP
        self.y += self.y_vel * self.TIMESTEP
        self.orbit.append((self.x, self.y))


def display_date():
    pass


def main():
    run = True
    clock = pygame.time.Clock() # Allows simulation to run at set speed rather than speed of computer

    # Initialize objects
    sun = Planet(0, 20, YELLOW, 1.98892e30)

    mecury = Planet(0.387, 3, DARK_GRAY, 3.3e23)
    venus = Planet(0.723, 7, ORANGE, 4.8685e24)
    earth = Planet(1, 8, LTBLUE, 5.9742e24)
    mars = Planet(1.524, 6, LTRED, 6.39e23)
    jupiter = Planet(5.2, 15, TURQUOISE, 1.89813e27)

    planets = [sun, mecury, venus, earth, mars, jupiter]

    while run:
        clock.tick(60) # Update at 60 FPS
        WINDOW.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        for planet in planets:
            if planet is not sun:
                planet.update_position(sun)
            planet.draw(WINDOW)

        pygame.display.update()

    pygame.quit()

main()
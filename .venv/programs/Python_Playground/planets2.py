import pygame
pygame.init()
import math
import random

WIDTH, HEIGHT = 1400, 1400
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Planet SimulationV2")

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

FONT = pygame.font.SysFont('verdana', 14, bold=True)
TIMESTEP_OPTIONS = [3600, 3600 * 6, 3600 * 12, 3600 * 24, 3600 * 24 * 2, 3600 * 24 * 7]
TIMESTEP_NAMES = ['One Hour', 'Six Hours', 'Tweleve Hours', 'One Day', 'Two Days', 'One Week']
TIMESTEP_INDEX = 3
TIMESTEP = TIMESTEP_OPTIONS[TIMESTEP_INDEX]
TIMESTEP_DISPLAY = TIMESTEP_NAMES[TIMESTEP_INDEX]


class Planet:
    AU = 149.6e9
    G = 6.67428e-11
    SCALE = 125 / AU 

    def __init__(self, name, distance_to_parent, radius, color, mass, *, parent=None, draw_orbit=True):
        self.name = name
        self.distance_to_parent = distance_to_parent
        self.radius = radius
        self.color = color
        self.mass = mass
        self.parent = parent
        self.draw_orbit = draw_orbit
        self.orbit = []
        self.x, self.y, self.x_vel, self.y_vel = self.random_start_location(distance_to_parent * Planet.AU)

    
    def random_start_location(self, distance):
        if self.parent:
            angle = 2 * math.pi * random.random()
            x = self.parent.x + distance * math.cos(angle)
            y = self.parent.y + distance * math.sin(angle)

            velocity = math.sqrt(Planet.G * self.parent.mass / distance)
            vy = velocity * math.cos(angle)

            vx = -velocity * math.sin(angle)

            if self.parent.parent:
                distance = math.sqrt(x**2 + y**2)
                theta = math.atan2(y, x)

                new_velocity = math.sqrt(Planet.G * self.parent.parent.mass / distance)
                vy += new_velocity * math.cos(theta)

                vx -= new_velocity * math.sin(theta)
            
        else:
            x = y = vx = vy = 0

        return x, y, vx, vy


    def draw(self, window, show_text=False):
        x = self.x * self.SCALE + WIDTH / 2
        y = self.y * self.SCALE + HEIGHT / 2

        if self.draw_orbit and len(self.orbit) > 2:
            updated_points = []
            for point in self.orbit:
                x, y = point
                x = x * self.SCALE + WIDTH / 2
                y = y * self.SCALE + HEIGHT / 2
                updated_points.append((x, y))

            pygame.draw.lines(window, self.color, False, updated_points, 2)

        pygame.draw.circle(window, self.color, (x, y), self.radius)
		
        if show_text and self.draw_orbit:
            # Draw name of object below object
            name_text = FONT.render(self.name, 1, WHITE)
            window.blit(name_text, (x - name_text.get_width()/2, y - name_text.get_height()/2 + self.radius + 10))
            # Draw distance from sun below name of object
            distance_text = FONT.render(f"{self.distance_to_parent:.4f}AU", 1, WHITE)
            window.blit(distance_text, (x - distance_text.get_width()/2, y - distance_text.get_height()/2 + self.radius + 30))


    def attraction(self):
        current = self.parent
        force_x = force_y = 0
        while current:
            distance_x = current.x - self.x
            distance_y = current.y - self.y
            distance = math.sqrt(distance_x**2 + distance_y**2)
            if current is self.parent:
                self.distance_to_parent = distance / Planet.AU

            force = self.G * self.mass * current.mass / distance ** 2
            theta = math.atan2(distance_y, distance_x)
            force_x += force * math.cos(theta)
            force_y += force * math.sin(theta)

            current = current.parent

        return force_x, force_y


    def update_position(self):
        if self.parent:
            fx, fy = self.attraction()

            self.x_vel += (fx / self.mass) * TIMESTEP
            self.y_vel += (fy / self.mass) * TIMESTEP

            self.x += self.x_vel * TIMESTEP
            self.y += self.y_vel * TIMESTEP
            self.orbit.append((self.x, self.y))


def create_asteroid_belt(planets: list, num_asteroids: int, sun: Planet) -> list:
    random_lst = [1] * 950 + [2] * 49 + [3] * 1

    for _ in range(num_asteroids):
        distance = random.gauss(2.67, 0.25)
        radius = random.choice(random_lst)
        mass = random.uniform(1e19, 1e21)
        planets.append(Planet('Asteroid', distance, radius, DARK_GRAY, mass, parent=sun, draw_orbit=False))
    
    return planets


class Toggle_Button(pygame.sprite.Sprite):
    def __init__(self, scale, x, y):
        super(Toggle_Button, self).__init__()
        self.scale = scale
        img_on = pygame.image.load('Python_Playground\\toggle1.png')
        self.img_on = pygame.transform.scale(img_on, self.scale)
        img_off = pygame.image.load('Python_Playground\\toggle2.png')
        self.img_off = pygame.transform.scale(img_off, self.scale)
        self.image = self.img_on
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.on = True


    def update(self, pos):
        if self.rect.collidepoint(pos):
            self.on = not self.on
            self.image = self.img_on if self.on else self.img_off

        return self.on


    def draw(self, win):
        Toggle_text = FONT.render("Toggle Text", 1, WHITE)
        win.blit(Toggle_text, (self.rect.x - Toggle_text.get_width() - 15, self.rect.y + 4))
        win.blit(self.image, self.rect)


def initialize_sim():
    # Initialize objects
    sun = Planet('Sun', 0, 20, YELLOW, 1.98892e30, draw_orbit=False)
    mecury = Planet('Mecury', 0.387, 4, DARK_GRAY, 3.3e23, parent=sun)
    venus = Planet('Venus', 0.723, 7, ORANGE, 4.8685e24, parent=sun)
    earth = Planet('Earth', 1, 8, LTBLUE, 5.9742e24, parent=sun)
    luna = Planet('Luna', .002569, 2, GREY, 7.34767309e22, parent=earth, draw_orbit=False)
    mars = Planet('Mars', 1.524, 5, LTRED, 6.39e23, parent=sun)
    jupiter = Planet('Jupiter', 5.2, 15, TURQUOISE, 1.89813e27, parent=sun)

    planets = [sun, mecury, venus, earth, luna, mars, jupiter]

    # Initilize asteroid belt
    planets = create_asteroid_belt(planets, 5000, sun)

    return planets



def display_date():
    pass


def increase_timestep():
    pass


def decrease_timestep():
    pass


def main():
    clock = pygame.time.Clock() # Allows simulation to run at set speed rather than speed of computer

    planets = initialize_sim()

    # Text toggle button
    toggle1_btn = Toggle_Button((50,30), 1325, 25)
    show_text = True

    while True:
        clock.tick(60) # Update at 60 FPS
        WINDOW.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # Toggle text button
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                show_text = toggle1_btn.update(pos)

            # Restart simulation
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                planets = initialize_sim()

        for planet in planets:
            planet.update_position()
            planet.draw(WINDOW, show_text)

        toggle1_btn.draw(WINDOW)
        pygame.display.update()


if __name__ == "__main__":
    main()


"""TODO
Make button not look terrible
get moons to work
Display simulated date
adjustable timestep
"""
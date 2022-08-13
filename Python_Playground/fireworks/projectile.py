import pygame
import random
import math
from particle import Particle
from settings import GRAVITY, FPS
from colors import BROWN

TRAIL_DENSITY_LIGHT = 8
TRAIL_DENSITY_MED = 5
TRAIL_DENSITY_HEAVY = 2

TRAIL_INITIAL_VEL_DECR = 0.8
SHELL_SIZE = 2
BURST_PARTICLE_INITIAL_VEL_DECR = 0.5

SHELL_PROPERTIES = {
    'small': (100, 2, TRAIL_DENSITY_LIGHT),
    'medium': (250, 2, TRAIL_DENSITY_LIGHT), 
    'large': (500, 3, TRAIL_DENSITY_MED),
    'extra_large': (750, 3, TRAIL_DENSITY_MED),
    'super': (1000, 4, TRAIL_DENSITY_HEAVY),
    'enormous': (2000, 4, TRAIL_DENSITY_HEAVY)
}

FLIGHT_PROFILE = {
    'high': (random.randint(-2, 2), -20 + random.randint(-2, 2), 115 + random.randint(-5, 5)),
    'med_high': (random.randint(-2, 2), -17 + random.randint(-2, 2), 105 + random.randint(-5, 5)),
    'medium': (random.randint(-2, 2), -15 + random.randint(-2, 2), 90 + random.randint(-5, 5)),
    'low': (random.randint(-2, 2), -10 + random.randint(-2, 2), 60 + random.randint(-5, 5))
}


class Projectile:
    def __init__(self, x: int, y: int, profile: str, trail_color: tuple=None, burst_size: str='large', *particle_colors: tuple):
        self.x = x
        self.y = y
        self.vel = pygame.Vector2(FLIGHT_PROFILE[profile][0], FLIGHT_PROFILE[profile][1])
        self.fuse = FLIGHT_PROFILE[profile][2]
        self.trail_color = trail_color
        self.num_burst_particles = SHELL_PROPERTIES[burst_size][0]
        self.particle_mass = SHELL_PROPERTIES[burst_size][1]
        self.trail_density = SHELL_PROPERTIES[burst_size][2]
        self.particle_colors = particle_colors

    @property
    def timeout(self):
        return self.fuse < 0

    @property
    def rand_color(self):
        return random.choice(self.particle_colors)

    def update(self):
        self.fuse -= 1
        self.x += self.vel.x
        self.y += self.vel.y
        self.vel.y += GRAVITY / FPS
        if self.trail_color and self.fuse % self.trail_density == 0:
            return Particle(self.x, self.y, self.vel.x * TRAIL_INITIAL_VEL_DECR + random.randint(-1, 1), self.vel.y * TRAIL_INITIAL_VEL_DECR + random.randint(-2, 2), random.uniform(0.25, self.particle_mass), self.trail_color)

    def burst(self):
        particles = []
        for _ in range(self.num_burst_particles):
            part_angle = random.uniform(0, 2 * math.pi)
            vel_mag = random.uniform(1, 20)
            particles.append(Particle(self.x, self.y, self.vel.x * BURST_PARTICLE_INITIAL_VEL_DECR + vel_mag * math.cos(part_angle), self.vel.y * BURST_PARTICLE_INITIAL_VEL_DECR + vel_mag * math.sin(part_angle), random.uniform(0, self.particle_mass), self.rand_color))

        return particles

    def draw(self, surface):
        pygame.draw.circle(surface, BROWN, (self.x, self.y), SHELL_SIZE)

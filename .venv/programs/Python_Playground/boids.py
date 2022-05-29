import pygame
pygame.init()
import random
import math
from p5 import Vector
import numpy as np


# Initialize the main window
NUM_BOIDS = 50
PERCEPTION_RADIUS = 100
MAX_FORCE = 0.5
MAX_SPEED = 5
WIDTH = pygame.display.get_desktop_sizes()[0][0]
HEIGHT = pygame.display.get_desktop_sizes()[0][1]
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boids")
FPS = 60

# Colors
GREEN = (0, 255, 0)
YELLOW = (220, 220, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
L_GRAY = (200, 200, 200)
D_GRAY = (70, 70, 70)


class Boid:
    def __init__(self):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        self.location = Vector(x, y)
        vec = (np.random.rand(2) - 0.5)*10
        self.velocity = Vector(*vec)
        vec = (np.random.rand(2) - 0.5)*10
        self.acceleration = Vector(*vec)
        self.color = (0, random.randint(0, 125), 255)


    def draw(self, window):
        pygame.draw.circle(window, self.color, (int(self.location.x), int(self.location.y)), 5, 0)


    def check_out_of_bounds(self):
        if self.location.x < 0:
            self.location.x = WIDTH
        elif self.location.x > WIDTH:
            self.location.x = 0
        if self.location.y < 0:
            self.location.y = HEIGHT
        elif self.location.y > HEIGHT:
            self.location.y = 0


    def update(self):
        self.location += self.velocity
        self.check_out_of_bounds()
        self.velocity += self.acceleration
        self.velocity.limit(MAX_SPEED)


    def align(self, boids):
        steering = Vector(*np.zeros(2))
        total = 0
        avg_vec = Vector(*np.zeros(2))
        for boid in boids:
            if np.linalg.norm(boid.location - self.location) < PERCEPTION_RADIUS:
                avg_vec += boid.velocity
                total += 1
        if total > 0:
            avg_vec /= total
            avg_vec = Vector(*avg_vec)
            avg_vec = (avg_vec /np.linalg.norm(avg_vec)) * MAX_SPEED
            steering = avg_vec - self.velocity
            steering.limit(MAX_FORCE)

        self.acceleration += steering


    def cohesion(self, boids):
        pass


    def seperation(self, boids):
        pass


class Simulation:
    def __init__(self, window):
        self.window = window
        self.boids = []
        for _ in range(NUM_BOIDS):
            boid = Boid()
            self.boids.append(boid)

    
    def draw(self):
        self.window.fill(BLACK)
        for boid in self.boids:
            boid.draw(self.window)

        pygame.display.update()


    def update(self):
        for boid in self.boids:
            boid.align(self.boids)
            boid.cohesion(self.boids)
            boid.seperation(self.boids)
            boid.update()


def main():
    clock = pygame.time.Clock()
    boids = Simulation(WINDOW)

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()

                if event.key == pygame.K_r:
                    main()


        boids.update()
        boids.draw()


if __name__ == "__main__":
    main()
import pygame
import random
from p5 import Vector
import numpy as np
import multiprocessing
from itertools import repeat


# Initialize the main window
NUM_BOIDS = 50
PERCEPTION_RADIUS = 100
MAX_FORCE = 0.5
MAX_SPEED = 10
BOID_SIZE = 10
RANDOM_MAGNITUDE_RANGE = 0.3
ALIGNMENT_STRENGTH = 1
COHESION_STRENGTH = 1
SEPERATION_STRENGTH = 1
RANDOM_MAGNITUDE_MIN = 1 - RANDOM_MAGNITUDE_RANGE / 2
WIDTH = 3440
HEIGHT = 1440
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
        vec = (np.random.rand(2) - 0.5) * MAX_SPEED
        self.velocity = Vector(*vec)
        vec = (np.random.rand(2) - 0.5) * MAX_FORCE
        self.acceleration = Vector(*vec)
        self.color = (0, random.randint(0, 125), random.randint(200, 255))


    def draw(self, window):
        #pygame.draw.circle(window, self.color, (int(self.location.x), int(self.location.y)), 5, 0)
        normalized_velocity = self.velocity.normalize() * BOID_SIZE
        x, y = int(self.location.x), int(self.location.y)
        points = ((x + int(normalized_velocity.x), y + int(normalized_velocity.y)),
                  (x + normalized_velocity.y // 2, y - normalized_velocity.x // 2),
                  (x, y),
                  (x -  normalized_velocity.y // 2, y + normalized_velocity.y // 2))
        pygame.draw.polygon(window, self.color, (points), 0)


    def check_out_of_bounds(self):
        """Wrap boids to other edge of screen
        """
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


    def apply_behavior(self, boids):
        self.acceleration *= 0
        self._align(boids)
        self._cohesion(boids)
        self._seperation(boids)


    def _align(self, boids):
        """Steer to average direction of local flock

        Args:
            boids (list): List of boids in simulation
        """
        steering = Vector(*np.zeros(2))
        total = 0
        for boid in boids:
            if boid is not self and np.linalg.norm(boid.location - self.location) < PERCEPTION_RADIUS:
                steering += boid.velocity
                total += 1
        if total > 0:
            steering /= total
            steering.normalize()
            steering *= MAX_SPEED
            steering -= self.velocity
            steering.limit(MAX_FORCE)

        self.acceleration += steering


    def _cohesion(self, boids):
        """Steer boid to center of mass of local flock

        Args:
            boids (list): List of boids in simulation
        """
        steering = Vector(*np.zeros(2))
        total = 0
        for boid in boids:
            if boid is not self and np.linalg.norm(boid.location - self.location) < PERCEPTION_RADIUS:
                steering += boid.location
                total += 1
        if total > 0:
            steering /= total
            steering -= self.location
            steering.normalize()
            steering *= MAX_SPEED
            steering -= self.velocity
            steering.limit(MAX_FORCE)

        self.acceleration += steering


    def _seperation(self, boids):
        """Steer to avoid other boids

        Args:
            boids (list): List of boids in simulation
        """
        steering = Vector(*np.zeros(2))
        total = 0
        for boid in boids:
            distance = np.linalg.norm(boid.location - self.location)
            if boid is not self and distance < PERCEPTION_RADIUS:
                diff_vector = self.location - boid.location
                diff_vector /= distance ** 2
                steering += diff_vector
                total += 1
        if total > 0:
            steering /= total
            steering.normalize()
            steering *= MAX_SPEED
            steering -= self.velocity
            steering.limit(MAX_FORCE)

        self.acceleration += steering


    def efficient_behavior(self, boids):
        """Adjust Boid acceleration based on alignment, cohesion and seperation

        Args:
            boids (list): List of boids in simulation
        """
        # Keep some weight from previous accel
        self.acceleration /= 2

        # Initialize behavior vectors
        alignment_steering = Vector(*np.zeros(2))
        cohesion_steering = Vector(*np.zeros(2))
        seperation_steering = Vector(*np.zeros(2))

        num_close_boids = 0
        for boid in boids:
            distance = np.linalg.norm(boid.location - self.location)
            if boid is not self and distance < PERCEPTION_RADIUS:
                alignment_steering += boid.velocity
                cohesion_steering += boid.location
                diff_vector = self.location - boid.location
                diff_vector /= distance ** 2
                seperation_steering += diff_vector
                num_close_boids += 1

        if num_close_boids > 0:
            # Alignment
            alignment_steering /= num_close_boids
            alignment_steering.normalize()
            nudge(alignment_steering)
            alignment_steering *= MAX_SPEED
            alignment_steering -= self.velocity
            alignment_steering.limit(MAX_FORCE)

            # Cohesion
            cohesion_steering /= num_close_boids
            cohesion_steering -= self.location
            cohesion_steering.normalize()
            nudge(cohesion_steering)
            cohesion_steering *= MAX_SPEED
            cohesion_steering -= self.velocity
            cohesion_steering.limit(MAX_FORCE)

            # Seperation
            seperation_steering /= num_close_boids
            seperation_steering.normalize()
            nudge(seperation_steering)
            seperation_steering *= MAX_SPEED
            seperation_steering -= self.velocity
            seperation_steering.limit(MAX_FORCE)

        self.acceleration += ALIGNMENT_STRENGTH * alignment_steering + COHESION_STRENGTH * cohesion_steering + SEPERATION_STRENGTH * seperation_steering
        self.acceleration.limit(MAX_FORCE)


def nudge(vector):
    vector.x *= (random.random() % RANDOM_MAGNITUDE_RANGE + RANDOM_MAGNITUDE_MIN)
    vector.y *= (random.random() % RANDOM_MAGNITUDE_RANGE + RANDOM_MAGNITUDE_MIN)


class Flock:
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


    def update(self, pool):
        for boid in self.boids:
            boid.update()
            boid.efficient_behavior(self.boids[:])

        #pool.starmap(Boid.efficient_behavior, zip(self.boids, repeat(self.boids)))


def initialize_pygame():
    pygame.init()
    WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids")

    return WINDOW


def main():
    window = initialize_pygame()
    clock = pygame.time.Clock()
    flock = Flock(window)
    pool = multiprocessing.Pool(multiprocessing.cpu_count() -1)

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


        flock.update(pool)
        flock.draw()


if __name__ == "__main__":
    main()
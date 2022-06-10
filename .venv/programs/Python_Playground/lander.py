import pygame
from screeninfo import get_monitors


# Constants
monitor = get_monitors()[0]
WIDTH = monitor.width
HEIGHT = monitor.height
FPS = 60

# Colors
GREEN = (0, 255, 0)
YELLOW = (220, 220, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
L_GRAY = (200, 200, 200)
D_GRAY = (70, 70, 70)

class Lander:
    def __init__(self):
        pass

    def draw(self):
        pass


class Terrain:
    def __init__(self):
        pass

    def draw(self):
        pass


def draw(terrain, lander):
    terrain.draw()
    lander.draw()
    pygame.display.update()


def initialize_pygame():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ants")
    terrain = Terrain(WIDTH, HEIGHT, window)
    lander = Lander(WIDTH, HEIGHT, window)

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

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()

                elif event.key == pygame.K_r:
                    main()


        """flock.update(pool)
        flock.draw()"""


if __name__ == "__main__":
    main()
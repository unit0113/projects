import pygame
from cave_generator import Cave
from screeninfo import get_monitors


# Constants
WANDER_STRENGTH = 1
monitor = get_monitors()[0]
#WIDTH = monitor.width
#HEIGHT = monitor.height
WIDTH = HEIGHT = 1000
FPS = 60

# Colors
GREEN = (0, 255, 0)
YELLOW = (220, 220, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
L_GRAY = (200, 200, 200)
D_GRAY = (70, 70, 70)

class Ant:
    def __init__(self):
        pass



def draw(cave):
    cave.draw()
    pygame.display.update()


def initialize_pygame():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ants")
    cave = Cave(WIDTH, HEIGHT, window)

    return window, cave


def main():
    window, cave = initialize_pygame()
    clock = pygame.time.Clock()
    draw(cave)

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
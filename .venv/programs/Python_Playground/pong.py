import pygame
import random


# Constants
WANDER_STRENGTH = 1
WIDTH = 1000
HEIGHT = 1000
FPS = 60

# Colors
GREEN = (0, 255, 0)
YELLOW = (220, 220, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
L_GRAY = (200, 200, 200)
D_GRAY = (70, 70, 70)

class Paddle:
    def __init__(self, side, window):
        self.window = window
        self.length = HEIGHT // 10
        self.width = WIDTH // 75
        self.color = WHITE
        if side == 'left':
            self.x = self.width
        else:
            self.x = WIDTH - self.width * 2
        self.y = HEIGHT // 2 - self.length // 2
        self.rect = pygame.Rect(self.x, self.y, self.width, self.length)



    def draw(self):
        pygame.draw.rect(self.window, self.color, (self.x, self.y, self.width, self.length))


class Ball:
    def __init__(self, window):
        self.window = window
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.size = 10
        self.color = WHITE
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)


    def draw(self):
        pygame.draw.rect(self.window, self.color, (self.x, self.y, self.size, self.size))


class PongGame:
    def __init__(self, window):
        self.score = 0
        self.window = window
        self.left_paddle = Paddle('left', self.window)
        self.right_paddle = Paddle('right', self.window)
        self.ball = Ball(window)

        

    def draw(self):
        self.window.fill(BLACK)
        self.left_paddle.draw()
        self.right_paddle.draw()
        self.ball.draw()
        pygame.display.update()


def initialize_pygame():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong")

    return window


def main():
    window = initialize_pygame()
    pong = PongGame(window)
    clock = pygame.time.Clock()

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


        pong.draw()


if __name__ == "__main__":
    main()
import pygame
from cart import Cart

WIDTH, HEIGHT = 3440, 1440
FPS = 60
BACKGROUND_COLOR = (0, 0, 0)
BASE_COLOR = (25, 39, 52)


class Environment:
    def __init__(self, window):
        self.window = window
        base_height = 2 * HEIGHT // 3
        self.cart = Cart(window, base_height)
        self.base = pygame.Rect(0, base_height, WIDTH, HEIGHT - base_height)

    def cart_left(self):
        self.cart.left()

    def cart_right(self):
        self.cart.right()


    def update(self):
        
        self.window.fill(BACKGROUND_COLOR)
        pygame.draw.rect(self.window, BASE_COLOR, self.base)
        self.cart.draw()
        pygame.display.update()


def initialize_pygame():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Space Invaders")
    pygame.font.init()

    return window


def main():
    window = initialize_pygame()
    environment = Environment(window)
    clock = pygame.time.Clock()

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

        if keys[pygame.K_LEFT]:
            environment.cart_left()

        if keys[pygame.K_RIGHT]:
            environment.cart_right()

        environment.update()


if __name__ == "__main__":
    main()
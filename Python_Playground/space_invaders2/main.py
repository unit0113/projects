import pygame
import time

from src.settings import WIDTH, HEIGHT, FPS
from src.game import Game


def init_pygame() -> tuple[pygame.surface.Surface, pygame.time.Clock]:
    """Perform pygame initialization, create main window and clock

    Returns:
        tuple[pygame.surface.Surface, pygame.time.Clock]: main window and clock
    """

    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2145")

    return window, clock


def main() -> None:
    window, clock = init_pygame()
    game = Game()

    # Main loop
    prev_time = time.time()
    run = True

    while run:
        clock.tick(FPS)
        now = time.time()
        dt = now - prev_time
        prev_time = now

        # Event handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        inputs = pygame.key.get_pressed()
        # Quit if escape is pressed
        if inputs[pygame.K_ESCAPE]:
            pygame.quit()
            quit()

        # Reset if r is pressed
        if inputs[pygame.K_r]:
            main()

        game.update(dt)
        game.fire()
        game.draw(window)

        pygame.display.update()

    # Exit pygame
    pygame.quit()


if __name__ == "__main__":
    main()

import pygame
import time

from src.settings import WIDTH, HEIGHT, FPS
from src.game import Game


def main() -> None:
    """Initiates game loop"""
    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2145")
    game = Game()
    prev_time = time.time()
    run = True

    # Main loop
    while run:
        clock.tick(FPS)
        now = time.time()
        dt = now - prev_time
        prev_time = now

        # Event handler
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Quit if escape is pressed
                    pygame.quit()
                    quit()
                elif event.key == pygame.K_r:
                    game.reset()

        # Update game
        game.update(dt, events)
        game.draw(window)

        pygame.display.update()

    # Exit pygame
    pygame.quit()


if __name__ == "__main__":
    main()

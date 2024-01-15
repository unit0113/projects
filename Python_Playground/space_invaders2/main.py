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
    reset_presed = False

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

        inputs = pygame.key.get_pressed()
        # Quit if escape is pressed
        if inputs[pygame.K_ESCAPE]:
            pygame.quit()
            quit()

        # Reset if r is pressed
        if not reset_presed and inputs[pygame.K_r]:
            game.reset()
            reset_presed = True
        elif reset_presed and not inputs[pygame.K_r]:
            reset_presed = False

        # Update game
        game.update(dt, events)
        game.draw(window)

        pygame.display.update()

    # Exit pygame
    pygame.quit()


if __name__ == "__main__":
    main()

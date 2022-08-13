import pygame
from show_manager import Show_Manager
from settings import HEIGHT, WIDTH, FPS


def initialize_pygame():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fireworks")
    pygame.font.init()

    return window


def main():
    window = initialize_pygame()
    show_manager = Show_Manager(window)
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

        show_manager.update()
        show_manager.draw()


if __name__ == "__main__":
    main()

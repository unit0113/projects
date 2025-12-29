# https://github.com/argonautcode/animal-proc-anim/blob/main/Chain.pde
# https://www.youtube.com/watch?v=qlfh_rv6khY

import pygame

from snake import Snake


def main():
    pygame.init()
    display_surface = pygame.display.set_mode((1800, 1200))
    pygame.display.set_caption("Procedural Animation")
    clock = pygame.time.Clock()

    # Draw Object
    # obj = Worm()
    obj = Snake()

    while True:
        dt = clock.tick() / 1000
        display_surface.fill("black")

        # Event Loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        obj.update(dt, pygame.mouse.get_pos())
        obj.draw(display_surface)
        pygame.display.update()


if __name__ == "__main__":
    main()

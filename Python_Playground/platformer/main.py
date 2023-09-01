import pygame
import time

from src.settings import WIDTH, HEIGHT, FPS, TILE_SIZE
from src.world import World
from src.button import Button


def init_pygame() -> tuple[pygame.surface.Surface, pygame.time.Clock]:
    """ Perform pygame initialization, create main window and clock

    Returns:
        tuple[pygame.surface.Surface, pygame.time.Clock]: main window and clock
    """

    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Platformer')

    return window, clock


def load_images() -> dict[str: pygame.surface.Surface]:
    images = {}
    images['sun'] = pygame.image.load('assets/sun.png')
    images['sky'] = pygame.image.load('assets/sky.png')
    images['dirt'] = pygame.image.load('assets/dirt.png')
    images['grass'] = pygame.image.load('assets/grass.png')
    images['slime'] = pygame.image.load('assets/blob.png')
    images['lava'] = pygame.image.load('assets/lava.png')
    images['lava'] = pygame.transform.scale(images['lava'], (TILE_SIZE, TILE_SIZE // 2))
    images['death'] = pygame.image.load('assets/ghost.png')
    images['restart_btn'] = pygame.image.load('assets/restart_btn.png')

    player_sprites = {}
    player_sprites[0] = pygame.image.load('assets/guy1.png')
    player_sprites[0] = pygame.transform.scale(player_sprites[0], (40, 80))
    player_sprites[1] = pygame.image.load('assets/guy2.png')
    player_sprites[1] = pygame.transform.scale(player_sprites[1], (40, 80))
    player_sprites[2] = pygame.image.load('assets/guy3.png')
    player_sprites[2] = pygame.transform.scale(player_sprites[2], (40, 80))
    player_sprites[3] = pygame.image.load('assets/guy4.png')
    player_sprites[3] = pygame.transform.scale(player_sprites[3], (40, 80))

    return images, player_sprites


world_data = [
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 1], 
[1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 2, 2, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 7, 0, 5, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1], 
[1, 7, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 1], 
[1, 0, 2, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 2, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 2, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 2, 2, 2, 2, 1], 
[1, 0, 0, 0, 0, 0, 2, 2, 2, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1], 
[1, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]


def main() -> None:
    window, clock = init_pygame()
    images, player_sprites = load_images()
    world = World(images, player_sprites, window, world_data)
    restart_btn = Button(WIDTH // 2 - images['restart_btn'].get_width() // 2, HEIGHT // 2 - 100, images['restart_btn'], window)

    # Main loop
    run = True
    round_over = False
    prev_time = time.time()

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
            elif round_over and event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                if restart_btn.is_clicked(pos):
                    main()

        inputs = pygame.key.get_pressed()
        # Quit if escape is pressed
        if inputs[pygame.K_ESCAPE]:
            pygame.quit()
            quit()

        # Reset if r is pressed
        if inputs[pygame.K_r]:
            main()

        # Update world
        round_over = world.update(dt, inputs)

        # Draw world
        world.draw()

        # Draw restart buttons if round over
        if round_over:
            restart_btn.draw()  
            # Check button clicking
            if pygame.mouse.get_pressed()[0] == 1 and restart_btn.is_clicked(pygame.mouse.get_pos()):
                main()

        pygame.display.update()

    # Exit pygame
    pygame.quit()


if __name__ == '__main__':
    main()

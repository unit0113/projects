import pygame
from pygame import mixer
import time
from os import listdir

from src.settings import WIDTH, HEIGHT, FPS, TILE_SIZE, BLUE
from src.world import World
from src.button import Button
from src.game_state import GameState


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


def init_music():
    mixer.pre_init(44100, -16, 2, 512)
    mixer.init()


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
    images['start_btn'] = pygame.image.load('assets/start_btn.png')
    images['exit_btn'] = pygame.image.load('assets/exit_btn.png')
    images['gate'] = pygame.image.load('assets/exit.png')
    images['gate'] = pygame.transform.scale(images['gate'], (TILE_SIZE, int(TILE_SIZE * 1.5)))
    images['coin'] = pygame.image.load('assets/coin.png')
    images['coin'] = pygame.transform.scale(images['coin'], (TILE_SIZE // 2, TILE_SIZE // 2))
    images['platform'] = pygame.image.load('assets/platform.png')
    images['platform'] = pygame.transform.scale(images['platform'], (TILE_SIZE, TILE_SIZE // 2))

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


def load_sounds():
    sounds = {}
    sounds['coin'] = mixer.Sound('assets/coin.wav')
    sounds['coin'].set_volume(0.5)
    sounds['jump'] = mixer.Sound('assets/jump.wav')
    sounds['jump'].set_volume(0.5)
    sounds['game_over'] = mixer.Sound('assets/game_over.wav')
    sounds['game_over'].set_volume(0.5)

    return sounds


def draw_text(text: str, font: pygame.font.Font, color: tuple[int, int, int], x: int, y: int, window: pygame.surface.Surface) -> None:
        img = font.render(text, True, color)
        window.blit(img, (x - img.get_width() // 2, y - img.get_height() // 2))


def main(level: int) -> None:
    window, clock = init_pygame()
    images, player_sprites = load_images()
    sounds = load_sounds()
    world = World(images, sounds, player_sprites, window, level)
    restart_btn = Button(WIDTH // 2 - images['restart_btn'].get_width() // 2, HEIGHT // 2 - 100, images['restart_btn'], window)
    start_btn = Button(WIDTH // 2 - 350, HEIGHT // 2, images['start_btn'], window)
    exit_btn = Button(WIDTH // 2 +150, HEIGHT // 2, images['exit_btn'], window)
    game_over_font = pygame.font.SysFont('Baushaus 93', 70)

    run = False
    game_state = False
    prev_time = time.time()

    # Menu loop
    while not run:
        clock.tick(FPS)
        # Draw background
        world.draw_background()

        # Draw buttons
        start_btn.draw()
        exit_btn.draw()

        pygame.display.update()

        # Event handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Get button selections
        if pygame.mouse.get_pressed()[0] == 1:
            if start_btn.is_clicked(pygame.mouse.get_pos()):
                run = True
            elif exit_btn.is_clicked(pygame.mouse.get_pos()):
                pygame.quit()
                quit()            

    # Main loop
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
            world.score = 0
            level = 0
            world.reset(level)

        # Update world
        game_state = world.update(dt, inputs)

        # Draw world
        world.draw()

        # Draw restart buttons if round over
        if game_state == GameState.GAME_OVER:
            restart_btn.draw()
            draw_text('YOU DIED!', game_over_font, BLUE, WIDTH // 2, HEIGHT // 4, window)
            # Check button clicking
            if pygame.mouse.get_pressed()[0] == 1 and restart_btn.is_clicked(pygame.mouse.get_pos()):
                world.reset(level)

        # Advance if gate reached
        if game_state == GameState.ADVANCE:
            level = level + 1
            if level == len(listdir('levels')):
                restart_btn.draw()
                draw_text('YOU WIN!', game_over_font, BLUE, WIDTH // 2, HEIGHT // 4, window)
                # Check button clicking
                if pygame.mouse.get_pressed()[0] == 1 and restart_btn.is_clicked(pygame.mouse.get_pos()):
                    world.score = 0
                world.reset(0)
            else:
                world.reset(level)

        pygame.display.update()

    # Exit pygame
    pygame.quit()


if __name__ == '__main__':
    main(3)

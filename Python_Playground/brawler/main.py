import pygame
from pygame import mixer
import time
from math import ceil

from fighter import Fighter
from settings import WIDTH, HEIGHT, FPS, YELLOW, RED, WHITE, FIGHTER_1_HEALTH_BAR_X, FIGHTER_2_HEALTH_BAR_X, HEALTH_BAR_Y, FIGHTER_HEALTH, COUNTDOWN, GAME_OVER_TIME


def load_sprites(sprite_sheet: pygame.surface.Surface, num_steps: list[int], sprite_states: list[str], size: int, scale: int) -> dict[str: list]:
    """ Seperate a sprite sheet into a dict of seperate sprites

    Returns:
        dict[str: list]: sprite dict
    """

    x = 0
    y = 0
    sprites = {}
    for steps, state in zip(num_steps, sprite_states):
        sprites[state] = []
        x = 0
        for _ in range(steps):
            temp = sprite_sheet.subsurface(x, y, size, size)
            sprites[state].append(pygame.transform.scale(temp, (size * scale, size * scale)))
            x += size
        y += size

    return sprites


def init_pygame() -> tuple[pygame.surface.Surface, pygame.time.Clock]:
    """ Perform pygame initialization, create main window and clock

    Returns:
        tuple[pygame.surface.Surface, pygame.time.Clock]: main window and clock
    """

    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Brawler')

    return window, clock


def start_music() -> None:
    """ Loads and begins game music
    """

    mixer.init()
    pygame.mixer.music.load('assets/audio/music.mp3')
    pygame.mixer.music.set_volume(0.5)
    # Play music on repeat
    pygame.mixer.music.play(-1, 0.0, 5000)


def get_assets(sword_fx: pygame.mixer.Sound, magic_fx: pygame.mixer.Sound) -> tuple[pygame.surface.Surface, pygame.surface.Surface, Fighter, Fighter]:
    """ Loads main assets

    Args:
        sword_fx (pygame.mixer.Sound): weapon sound effects for warrior
        magic_fx (pygame.mixer.Sound): weapon sound effects for wizard

    Returns:
        tuple[pygame.surface.Surface, pygame.surface.Surface, Fighter, Fighter]: main assets
    """

    # Load images
    bg_img = pygame.image.load('assets/images/background/background.jpg').convert_alpha()
    bg_img = pygame.transform.scale(bg_img, (WIDTH, HEIGHT))
    warrior_sheet = pygame.image.load('assets/images/warrior/Sprites/warrior.png').convert_alpha()
    wizard_sheet = pygame.image.load('assets/images/wizard/Sprites/wizard.png').convert_alpha()

    # Number of frames in each sprite animation
    warrior_animation_steps = [10, 8, 1, 7, 7, 3, 7]
    wizard_animation_steps = [8, 8, 1, 8, 8, 3, 7]
    sprite_states = ['idle', 'run', 'jump', 'attack1', 'attack2', 'hit', 'death']

    # Size of each square on sprite sheet
    warrior_size = 162
    wizard_size = 250

    # Scaling factor
    warrior_scale = 4
    wizard_scale = 3

    # Offset amount to center image on rect
    warrior_offset = (72 * warrior_scale, 56 * warrior_scale)
    wizard_offset = (112 * wizard_scale, 106 * wizard_scale)

    # Get sprites
    warrior_sprites = load_sprites(warrior_sheet, warrior_animation_steps, sprite_states, warrior_size, warrior_scale)
    wizard_sprites = load_sprites(wizard_sheet, wizard_animation_steps, sprite_states, wizard_size, wizard_scale)

    # Create fighters
    fighter_1_controls = {'left': pygame.K_a, 'right': pygame.K_d, 'jump': pygame.K_w, 'attack1': pygame.K_q, 'attack2': pygame.K_e}
    fighter_2_controls = {'left': pygame.K_j, 'right': pygame.K_l, 'jump': pygame.K_i, 'attack1': pygame.K_u, 'attack2': pygame.K_o}
    fighter_1 = Fighter(200, 310, warrior_sprites, warrior_offset, fighter_1_controls, sword_fx)
    fighter_2 = Fighter(WIDTH - 300, 310, wizard_sprites, wizard_offset, fighter_2_controls, magic_fx)

    # Set enemies
    fighter_1.set_enemy(fighter_2)
    fighter_2.set_enemy(fighter_1)

    # Victory image
    victory_image = pygame.image.load('assets/images/icons/victory.png').convert_alpha()

    return bg_img, victory_image, fighter_1, fighter_2


def load_fonts() -> tuple[pygame.font.Font, pygame.font.Font]:
    """ Loads fonts

    Returns:
        tuple[pygame.font.Font, pygame.font.Font]: fonts
    """

    count_font = pygame.font.Font('assets/fonts/turok.ttf', 80)
    score_font = pygame.font.Font('assets/fonts/turok.ttf', 30)

    return count_font, score_font


def load_sound_effects() -> tuple[pygame.mixer.Sound, pygame.mixer.Sound]:
    """ Loads weapon sound effects

    Returns:
        tuple[pygame.mixer.Sound, pygame.mixer.Sound]: sound effects for fighters
    """

    sword_fx = pygame.mixer.Sound('assets/audio/sword.wav')
    sword_fx.set_volume(0.5)
    magic_fx = pygame.mixer.Sound('assets/audio/magic.wav')
    magic_fx.set_volume(0.75)

    return sword_fx, magic_fx

def draw_text(text: str, font: pygame.font.Font, text_col: tuple[int, int, int], x: int, y: int, window: pygame.surface.Surface) -> None:
    """ Draws text at specified location

    Args:
        text (str): Text to draw
        font (pygame.font.Font): Font to use
        text_col (tuple[int, int, int]): Text color
        x (int): X position
        y (int): Y position
        window (pygame.surface.Surface): Surface to draw on
    """

    img = font.render(text, True, text_col)
    window.blit(img, (x - img.get_width() // 2, y))


def draw_health_bars(fighter_1_hp: int, fighter_2_hp: int, window: pygame.surface.Surface) -> None:
    """ Draws fighter health bars

    Args:
        fighter_1_hp (int): Health of fighter
        fighter_2_hp (int): Health of fighter
        window (pygame.surface.Surface): Surface to draw on
    """
    
    # Fighter 1
    pygame.draw.rect(window, WHITE, (FIGHTER_1_HEALTH_BAR_X - 2, HEALTH_BAR_Y - 2, 404, 34))
    pygame.draw.rect(window, RED, (FIGHTER_1_HEALTH_BAR_X, HEALTH_BAR_Y, 400, 30))
    pygame.draw.rect(window, YELLOW, (FIGHTER_1_HEALTH_BAR_X, HEALTH_BAR_Y, 400 * (fighter_1_hp / FIGHTER_HEALTH), 30))

    # Fighter 2
    pygame.draw.rect(window, WHITE, (FIGHTER_2_HEALTH_BAR_X - 2, HEALTH_BAR_Y - 2, 404, 34))
    pygame.draw.rect(window, RED, (FIGHTER_2_HEALTH_BAR_X, HEALTH_BAR_Y, 400, 30))
    pygame.draw.rect(window, YELLOW, (FIGHTER_2_HEALTH_BAR_X, HEALTH_BAR_Y, 400 * (fighter_2_hp / FIGHTER_HEALTH), 30))


def main(score: list[int, int]= [0, 0]) -> None:
    score = score
    window, clock = init_pygame()
    sword_fx, magic_fx = load_sound_effects()
    bg_img, victory_image, fighter_1, fighter_2 = get_assets(sword_fx, magic_fx)
    count_font, score_font = load_fonts()
    round_over = False
    countdown = COUNTDOWN
    game_over_timer = 0

    # Main loop
    run = True
    prev_time = time.time()

    while run:
        clock.tick(FPS)
        now = time.time()
        dt = now - prev_time
        prev_time = now
        countdown -= dt

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
            main([0,0])
        
        # Draw background
        window.blit(bg_img, (0,0))

        # Update fighters
        fighter_1.update(dt)
        fighter_2.update(dt)

        if countdown < 0:
            fighter_1.move(inputs)
            fighter_2.move(inputs)
        else:
            draw_text(str(ceil(countdown)), count_font, RED, WIDTH // 2, HEIGHT // 4, window)

        # Draw assets
        fighter_1.draw(window)
        fighter_2.draw(window)

        # Draw health bars
        draw_health_bars(fighter_1.curr_hp, fighter_2.curr_hp, window)

        # Draw scores
        draw_text(str(score[0]), score_font, RED, 25, 50, window)
        draw_text(str(score[1]), score_font, RED, 975, 50, window)

        # Check for round over
        if not round_over:
            if fighter_1.game_over:
                score[1] += 1
                round_over = True
            elif fighter_2.game_over:
                score[0] += 1
                round_over = True
        else:
            game_over_timer += dt
            if game_over_timer > GAME_OVER_TIME:
                main(score)
            window.blit(victory_image, (WIDTH // 2 - victory_image.get_width() // 2, 150))

        pygame.display.update()

    # Exit pygame
    pygame.quit()


if __name__ == '__main__':
    start_music()
    main()

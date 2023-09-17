import pygame
import time
import os

from settings import WIDTH, HEIGHT, FPS, CASTLE_X, CASTLE_Y, PRICE_SCALE, TOWER_PRICE, REPAIR_PRICE, ARMOR_PRICE
from castle import Castle
from crosshair import Crosshair
from enemy_manager import EnemyManager
from button import Button

CASTLE_SCALE = 0.2
TOWER_SCALE = 0.2
TOWER_BUTTON_SCALE = 0.5
BULLET_SCALE = 0.075
ENEMY_SCALE = 0.2
CROSSHAIR_SCALE = 0.025
REPAIR_SCALE = 0.5
ARMOR_SCALE = 1.5
END_LEVEL_DELAY = 1.5
TOWER_LOCATIONS = [(WIDTH - 250, HEIGHT - 200), (WIDTH - 200, HEIGHT - 150), (WIDTH - 150, HEIGHT - 150), (WIDTH - 100, HEIGHT - 150)]

pygame.font.init()
S_FONT = pygame.font.SysFont('Futura', 30)
L_FONT = pygame.font.SysFont('Futura', 60)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)


def draw_text(window: pygame.surface.Surface, text: str, font: pygame.font.SysFont, text_col: tuple[int, int, int], x: int, y: int, *, center_x=True) -> None:
    img = font.render(text, True, text_col)
    if center_x:
        window.blit(img, (x - img.get_width() // 2, y))
    else:
        window.blit(img, (x, y))


def init_pygame() -> tuple[pygame.surface.Surface, pygame.time.Clock]:
    """ Perform pygame initialization, create main window and clock

    Returns:
        tuple[pygame.surface.Surface, pygame.time.Clock]: main window and clock
    """

    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Castle Defender')

    return window, clock


def load_assets() -> dict[str, pygame.surface.Surface]:
    assets = {}
    assets['background'] = pygame.image.load('assets/bg.png').convert_alpha()
    assets['castle_100'] = pygame.image.load('assets/castle/castle_100.png').convert_alpha()
    assets['castle_100'] = pygame.transform.scale(assets['castle_100'], (int(assets['castle_100'].get_width() * CASTLE_SCALE), int(assets['castle_100'].get_height() * CASTLE_SCALE)))
    assets['castle_50'] = pygame.image.load('assets/castle/castle_50.png').convert_alpha()
    assets['castle_50'] = pygame.transform.scale(assets['castle_50'], (int(assets['castle_50'].get_width() * CASTLE_SCALE), int(assets['castle_50'].get_height() * CASTLE_SCALE)))
    assets['castle_25'] = pygame.image.load('assets/castle/castle_25.png').convert_alpha()
    assets['castle_25'] = pygame.transform.scale(assets['castle_25'], (int(assets['castle_25'].get_width() * CASTLE_SCALE), int(assets['castle_25'].get_height() * CASTLE_SCALE)))
    assets['bullet'] = pygame.image.load('assets/bullet.png').convert_alpha()
    assets['bullet'] = pygame.transform.scale(assets['bullet'], (int(assets['bullet'].get_width() * BULLET_SCALE), int(assets['bullet'].get_height() * BULLET_SCALE)))
    assets['crosshair'] = pygame.image.load('assets/crosshair.png').convert_alpha()
    assets['crosshair'] = pygame.transform.scale(assets['crosshair'], (int(assets['crosshair'].get_width() * CROSSHAIR_SCALE), int(assets['crosshair'].get_height() * CROSSHAIR_SCALE)))
    tower = {}
    tower['tower_100'] = pygame.image.load('assets/tower/tower_100.png').convert_alpha()
    tower['tower_100'] = pygame.transform.scale(tower['tower_100'], (int(tower['tower_100'].get_width() * TOWER_SCALE), int(tower['tower_100'].get_height() * TOWER_SCALE)))
    tower['tower_50'] = pygame.image.load('assets/tower/tower_50.png').convert_alpha()
    tower['tower_50'] = pygame.transform.scale(tower['tower_50'], (int(tower['tower_50'].get_width() * TOWER_SCALE), int(tower['tower_50'].get_height() * TOWER_SCALE)))
    tower['tower_25'] = pygame.image.load('assets/tower/tower_25.png').convert_alpha()
    tower['tower_25'] = pygame.transform.scale(tower['tower_25'], (int(tower['tower_25'].get_width() * TOWER_SCALE), int(tower['tower_25'].get_height() * TOWER_SCALE)))
    tower['tower_button'] = pygame.transform.scale(tower['tower_100'], (int(tower['tower_100'].get_width() * TOWER_BUTTON_SCALE), int(tower['tower_100'].get_height() * TOWER_BUTTON_SCALE)))
    assets['tower'] = tower
    
    # Load enemy sprites
    enemy_types = ['goblin', 'knight', 'purple_goblin', 'red_goblin']
    animation_types = ['walk', 'attack', 'death']
    for enemy in enemy_types:
        assets[enemy] = {}
        for animation in animation_types:
            sprites = []
            num_animations = len(os.listdir(f'assets\enemies\{enemy}\{animation}'))
            for index in range(num_animations):
                try:
                    img = pygame.image.load(f'assets\enemies\{enemy}\{animation}\{index}.png').convert_alpha()
                    sprites.append(pygame.transform.scale(img, (int(img.get_width() * ENEMY_SCALE), int(img.get_height() * ENEMY_SCALE))))
                except:
                    pass
            assets[enemy][animation] = sprites[:]

    # Load buttons
    buttons = {}
    buttons['repair'] = pygame.image.load('assets/repair.png').convert_alpha()
    buttons['repair'] = pygame.transform.scale(buttons['repair'], (int(buttons['repair'].get_width() * REPAIR_SCALE), int(buttons['repair'].get_height() * REPAIR_SCALE)))
    buttons['armor'] = pygame.image.load('assets/armour.png').convert_alpha()
    buttons['armor'] = pygame.transform.scale(buttons['armor'], (int(buttons['armor'].get_width() * ARMOR_SCALE), int(buttons['armor'].get_height() * ARMOR_SCALE)))

    assets['buttons'] = buttons

    return assets


def display_game_info(window: pygame.surface.Surface, money: int, score: int, high_score: int, level: int, health_status: tuple[int, int], tower_price: int, armor_price: int) -> None:
    draw_text(window, f'Money: {money}', S_FONT, GRAY, 10, 10, center_x=False)
    draw_text(window, f'Score: {score}', S_FONT, GRAY, 180, 10, center_x=False)
    draw_text(window, f'High Score: {high_score}', S_FONT, GRAY, 180, 30, center_x=False)
    draw_text(window, f'Level: {level}', S_FONT, GRAY, WIDTH // 2, 10)
    draw_text(window, f'Health: {health_status[0]} / {health_status[1]}', S_FONT, GRAY, WIDTH - 230, HEIGHT - 50, center_x=False)

    # Draw prices
    draw_text(window, '1000', S_FONT, GRAY, WIDTH - 220, 70, center_x=False)
    draw_text(window, f'{tower_price}', S_FONT, GRAY, WIDTH - 140, 70, center_x=False)
    draw_text(window, f'{armor_price}', S_FONT, GRAY, WIDTH - 75, 70, center_x=False)


def load_high_score() -> int:
    path = 'score.txt'
    high_score = 0
    if os.path.isfile(path):
        with open(path, 'r') as file:
            high_score = int(file.read())

    return high_score


def save_high_score(score: int) -> None:
    with open('score.txt', 'w') as file:
        file.write(str(score))


def main() -> None:
    window, clock = init_pygame()
    assets = load_assets()
    castle = Castle(assets, CASTLE_X, CASTLE_Y)
    crosshair = Crosshair(assets['crosshair'])
    enemy_manager = EnemyManager(assets)
    repair_button = Button(WIDTH - 220, 10, assets['buttons']['repair'])
    tower_button = Button(WIDTH - 140, 10, assets['tower']['tower_button'])
    armor_button = Button(WIDTH - 75, 10, assets['buttons']['armor'])

    game_over = False
    money = 0
    score = 0
    high_score = load_high_score()
    level_end_timer = 0
    level_counter = 1
    tower_count = 0
    tower_price = TOWER_PRICE
    armor_price = ARMOR_PRICE

    # Main loop
    run = True
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

        inputs = pygame.key.get_pressed()
        # Quit if escape is pressed
        if inputs[pygame.K_ESCAPE]:
            pygame.quit()
            quit()

        # Reset if r is pressed
        if inputs[pygame.K_r]:
            castle.reset()
            enemy_manager.reset()
            game_over = False
            money = 0
            score = 0
            level_end_timer = 0
            level_counter = 1
            tower_count = 0
            tower_price = TOWER_PRICE
            armor_price = ARMOR_PRICE

        if game_over:
            draw_text(window, 'GAME OVER!', L_FONT, GRAY, 300, 300)
            draw_text(window, 'Press R to Restart', L_FONT, GRAY, 250, 360)
            pygame.mouse.set_visible(True)
            pygame.display.update()
            continue

        # Update
        castle.update(dt)
        game_over = castle.take_damage(enemy_manager.damage_castle())
        
        # Draw background
        window.blit(assets['background'], (0,0))

        # Draw
        castle.draw(window)
        enemy_manager.update(dt, window)
        new_points = enemy_manager.take_bullets(castle.bullet_group)
        money += new_points
        score += new_points

        crosshair.draw(window)
        
        # Check level complete
        if enemy_manager.level_complete():
            draw_text(window, f'LEVEL {level_counter} COMPLETE!', L_FONT, WHITE, WIDTH // 2, HEIGHT // 2)
            level_end_timer += dt
            if level_end_timer > END_LEVEL_DELAY:
                enemy_manager.next_level()
                level_end_timer = 0
                level_counter += 1
                if score > high_score:
                    high_score = score
                    save_high_score(high_score)

        # Display stats
        display_game_info(window, money, score, high_score, level_counter, castle.get_health_status(), tower_price, armor_price)

        # Display buttons and perform button actions
        if (repair_button.draw(window) or inputs[pygame.K_a]) and money > REPAIR_PRICE and castle.can_repair():
            castle.repair()
            money -= REPAIR_PRICE

        if (tower_button.draw(window) or inputs[pygame.K_s]) and money > tower_price and tower_count < len(TOWER_LOCATIONS):
            castle.add_tower(*TOWER_LOCATIONS[tower_count])
            tower_count += 1
            money -= tower_price
            tower_price = int(tower_price * PRICE_SCALE)

        if (armor_button.draw(window) or inputs[pygame.K_d]) and money > armor_price:
            castle.upgrade_armor()
            money -= armor_price
            armor_price = int(armor_price * PRICE_SCALE)

        pygame.display.update()

    # Exit pygame
    pygame.quit()


if __name__ == '__main__':
    main()

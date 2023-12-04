import pygame
import time
import os
import json

from src.settings import WIDTH, HEIGHT, FPS, SIDE_PANEL_WIDTH, TILESIZE
from src.enemy import Enemy
from src.world import World
from src.turret import Turret
from src.button import Button
from src.enemy_factory import EnemyFactory


def init_pygame() -> tuple[pygame.surface.Surface, pygame.time.Clock]:
    """Perform pygame initialization, create main window and clock

    Returns:
        tuple[pygame.surface.Surface, pygame.time.Clock]: main window and clock
    """

    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode((WIDTH + SIDE_PANEL_WIDTH, HEIGHT))
    pygame.display.set_caption("Tower Defense")

    return window, clock


def load_assets() -> dict[str : pygame.surface.Surface]:
    assets = {}
    path = "src/assets/images"

    # Load enemies
    enemy_images = {}
    strength_levels = ["weak", "medium", "strong", "elite"]
    folder = path + "/enemies"
    for strength, enemy in zip(strength_levels, os.listdir(folder)):
        enemy_images[strength] = pygame.image.load(f"{folder}/{enemy}").convert_alpha()
    assets["enemies"] = enemy_images

    # Load levels
    folder = "src/assets/levels"
    for level in os.listdir(folder):
        name, extension = level.split(".")
        if extension == "png":
            assets[name] = pygame.image.load(f"{folder}/{level}").convert_alpha()
        else:
            with open(f"{folder}/{level}") as file:
                level_data = json.load(file)
                assets[f"{name}_data"] = level_data

    # Load turrets
    folder = path + "/turrets"
    turrets = []
    for file in os.listdir(folder):
        if file.split(".")[0] == "cursor_turret":
            assets[file.split(".")[0]] = pygame.image.load(
                f"{folder}/{file}"
            ).convert_alpha()
        else:
            turrets.append(pygame.image.load(f"{folder}/{file}").convert_alpha())
    assets["turrets"] = turrets

    # Load buttons
    folder = path + "/buttons"
    for file in os.listdir(folder):
        assets[f'button_{file.split(".")[0]}'] = pygame.image.load(
            f"{folder}/{file}"
        ).convert_alpha()

    return assets


def main() -> None:
    window, clock = init_pygame()
    assets = load_assets()
    world = World(assets["level0"], assets["level0_data"], assets["enemies"])

    # Create groups
    enemy_group = pygame.sprite.Group()
    turret_group = pygame.sprite.Group()
    selected_turret = None

    # Create Buttons
    turret_button = Button((WIDTH + 30, 120), assets["button_buy_turret"], True)
    cancel_button = Button((WIDTH + 50, 180), assets["button_cancel"], True)
    upgrade_button = Button((WIDTH + 5, 180), assets["button_upgrade_turret"], True)
    begin_button = Button((WIDTH + 60, 300), assets["button_begin"], True)

    # Main loop
    prev_time = time.time()
    run = True
    is_placing_turrets = False
    level_started = False

    while run:
        clock.tick(FPS)
        now = time.time()
        dt = now - prev_time
        prev_time = now

        # Draw Backgrounds
        window.fill("grey100")
        world.draw(window)

        # Event handler
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if selected_turret and mouse_pos[0] < WIDTH:
                    selected_turret.selected = False
                    selected_turret = None
                # Check if mouse pos is valid
                if is_placing_turrets and world.valid_turret_location(mouse_pos):
                    if world.purchase_new_turret():
                        turret_group.add(Turret(assets["turrets"], mouse_pos))
                    else:
                        world.reset_turret_location(mouse_pos)
                elif not is_placing_turrets:
                    # Select turret
                    mouse_tile_x = mouse_pos[0] // TILESIZE
                    mouse_tile_y = mouse_pos[1] // TILESIZE
                    for turret in turret_group:
                        if (mouse_tile_x, mouse_tile_y) == (
                            turret.tile_x,
                            turret.tile_y,
                        ):
                            selected_turret = turret
                            break

        inputs = pygame.key.get_pressed()
        # Quit if escape is pressed
        if inputs[pygame.K_ESCAPE]:
            pygame.quit()
            quit()

        # Reset if r is pressed
        if inputs[pygame.K_r]:
            main()

        # Draw buttons
        if turret_button.draw(window):
            is_placing_turrets = True
        elif is_placing_turrets:
            cursor_turret_rect = assets["cursor_turret"].get_rect()
            if mouse_pos[0] < WIDTH:
                cursor_turret_rect.center = mouse_pos
                window.blit(assets["cursor_turret"], cursor_turret_rect)

            if cancel_button.draw(window):
                is_placing_turrets = False

        elif selected_turret:
            if (
                selected_turret.can_upgrade()
                and upgrade_button.draw(window)
                and world.purchase_turret_upgrade()
            ):
                selected_turret.upgrade()

        if not level_started:
            level_started = begin_button.draw(window)
            if level_started:
                world.new_level()
                enemy_group = world.get_enemies()

        else:
            # Update groups
            enemy_group.update(world)
            new_money = sum([turret.update(enemy_group) for turret in turret_group])
            world.updateMoney(new_money)

        if selected_turret:
            selected_turret.selected = True

        # Draw path
        pygame.draw.lines(window, "grey0", False, world.waypoints)

        # Draw groups
        enemy_group.draw(window)
        # turret_group.draw(window)
        for turret in turret_group:
            turret.draw(window)

        # Draw UI
        world.draw_ui(window)

        pygame.display.update()

        if level_started and len(enemy_group) == 0:
            level_started = False

    # Exit pygame
    pygame.quit()


if __name__ == "__main__":
    main()

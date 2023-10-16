import pygame
import time
import os
import random
from enum import Enum

from settings import WIDTH, PANEL_HEIGHT, HEIGHT, FPS, BANDIT_HEAL_CHANCE
from fighter import Fighter
from button import Button


RED = (255, 0, 0)
GREEN = (0, 255, 0)
HEALTH_BAR_WIDTH = 150
HEALTH_BAR_HEIGHT = 20
ACTION_COOLDOWN = 0.5


class GameState(Enum):
    run = 1
    player_win = 2
    player_loss = 3
    quit = 4


class DamageText(pygame.sprite.Sprite):
    def __init__(self, x, y, damage, color) -> None:
        pygame.sprite.Sprite.__init__(self)
        font = pygame.font.SysFont("Times New Roman", 26)
        self.image = font.render(damage, True, color)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.counter = 0

    def update(self):
        self.rect.y -= 1
        self.counter += 1
        if self.counter > 30:
            self.kill()


def init_pygame() -> tuple[pygame.surface.Surface, pygame.time.Clock]:
    """Perform pygame initialization, create main window and clock

    Returns:
        tuple[pygame.surface.Surface, pygame.time.Clock]: main window and clock
    """

    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Brawler")

    return window, clock


def draw_text(window, text, color, x, y) -> None:
    draw_text.font = getattr(
        draw_text, "font", pygame.font.SysFont("Times New Roman", 26)
    )
    img = draw_text.font.render(text, True, color)
    window.blit(img, (x, y))


def draw_panel(window, panel_img, knight, bandits) -> None:
    window.blit(panel_img, (0, HEIGHT - PANEL_HEIGHT))
    # Draw knight panel
    draw_text(
        window, f"{knight.name} HP: {knight.hp}", RED, 100, HEIGHT - PANEL_HEIGHT + 10
    )
    draw_health_bar(window, 100, HEIGHT - PANEL_HEIGHT + 40, knight.hp, knight.max_hp)

    # Draw bandit panel
    for index, bandit in enumerate(bandits):
        draw_text(
            window,
            f"{bandit.name} HP: {bandit.hp}",
            RED,
            550,
            HEIGHT - PANEL_HEIGHT + 10 + index * 60,
        )
        draw_health_bar(
            window,
            550,
            HEIGHT - PANEL_HEIGHT + 40 + index * 60,
            bandit.hp,
            bandit.max_hp,
        )


def draw_health_bar(window, x, y, hp, max_hp) -> None:
    pygame.draw.rect(window, RED, (x, y, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT))
    pygame.draw.rect(
        window, GREEN, (x, y, int(HEALTH_BAR_WIDTH * (hp / max_hp)), HEALTH_BAR_HEIGHT)
    )


def load_images() -> dict:
    assets = {}
    assets["bg"] = pygame.image.load("assets/Background/background.png").convert_alpha()
    assets["panel"] = pygame.image.load("assets/Icons/panel.png").convert_alpha()
    assets["sword"] = pygame.image.load("assets/Icons/sword.png").convert_alpha()
    assets["potion"] = pygame.image.load("assets/Icons/potion.png").convert_alpha()
    assets["victory"] = pygame.image.load("assets/Icons/victory.png").convert_alpha()
    assets["defeat"] = pygame.image.load("assets/Icons/defeat.png").convert_alpha()
    assets["restart"] = pygame.image.load("assets/Icons/restart.png").convert_alpha()

    # Load fighter sprites
    types = ["Knight", "Bandit"]
    animation_types = ["Attack", "Death", "Hurt", "Idle"]
    for fighter in types:
        assets[fighter] = {}
        for animation in animation_types:
            sprites = []
            num_animations = len(os.listdir(f"assets\{fighter}\{animation}"))
            for index in range(num_animations):
                try:
                    img = pygame.image.load(
                        f"assets\{fighter}\{animation}\{index}.png"
                    ).convert_alpha()
                    sprites.append(
                        pygame.transform.scale(
                            img, (img.get_width() * 3, img.get_height() * 3)
                        )
                    )
                except:
                    break
            assets[fighter][animation] = sprites[:]

    return assets


def main() -> None:
    window, clock = init_pygame()
    assets = load_images()
    player = Fighter("Knight", 200, 260, 30, 10, 3, assets["Knight"])
    bandit_list = [
        Fighter("Bandit", 550, 270, 20, 6, 1, assets["Bandit"]),
        Fighter("Bandit", 700, 270, 20, 6, 1, assets["Bandit"]),
    ]
    action_timer = 0
    current_fighter = 0
    potion_button = Button(
        window, 100, HEIGHT - PANEL_HEIGHT + 70, assets["potion"], 64, 64
    )
    restart_button = Button(window, 330, 120, assets["restart"], 120, 30)

    damage_text_group = pygame.sprite.Group()

    # Main loop
    game_state = GameState.run
    prev_time = time.time()

    while game_state != GameState.quit:
        clock.tick(FPS)
        now = time.time()
        dt = now - prev_time
        prev_time = now
        action_timer += dt

        # Event handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                clicked = True
            else:
                clicked = False

        inputs = pygame.key.get_pressed()
        # Quit if escape is pressed
        if inputs[pygame.K_ESCAPE]:
            pygame.quit()
            quit()

        # Reset if r is pressed
        if inputs[pygame.K_r]:
            main()

        # Draw background
        window.blit(assets["bg"], (0, 0))
        draw_panel(window, assets["panel"], player, bandit_list)

        # Use potion
        use_potion = potion_button.draw() and player.can_use_potion
        draw_text(window, str(player.potions), RED, 150, HEIGHT - PANEL_HEIGHT + 70)

        # Update fighters
        player.update(dt)
        for bandit in bandit_list:
            bandit.update(dt)

        # Change mouse cursor if over bandit
        pos = pygame.mouse.get_pos()
        pygame.mouse.set_visible(True)
        target = None
        for bandit in bandit_list:
            if bandit.rect.collidepoint(pos):
                pygame.mouse.set_visible(False)
                window.blit(assets["sword"], pos)
                if clicked and bandit.alive:
                    target = bandit

        # Perform action
        damage_text_group.update()

        if current_fighter == 0 and player.alive and action_timer > ACTION_COOLDOWN:
            if use_potion:
                healed = player.use_potion()
                damage_text_group.add(
                    DamageText(player.rect.centerx, player.rect.y, str(healed), GREEN)
                )

            elif target:
                dmg = player.attack(target)
                damage_text_group.add(
                    DamageText(target.rect.centerx, target.rect.y, str(dmg), RED)
                )

            if use_potion or target:
                current_fighter = 1 if bandit_list[0].alive else 2
                action_timer = 0

        elif (
            current_fighter == 1
            and bandit_list[0].alive
            and action_timer > ACTION_COOLDOWN
        ):
            # Bandit 1 action
            if bandit_list[0].should_heal and random.random() < BANDIT_HEAL_CHANCE:
                healed = bandit_list[0].use_potion()
                damage_text_group.add(
                    DamageText(
                        bandit_list[0].rect.centerx,
                        bandit_list[0].rect.y,
                        str(healed),
                        GREEN,
                    )
                )

            else:
                dmg = bandit_list[0].attack(player)
                damage_text_group.add(
                    DamageText(player.rect.centerx, player.rect.y, str(dmg), RED)
                )

            current_fighter = 2 if bandit_list[1].alive else 0
            action_timer = 0

        elif (
            current_fighter == 2
            and bandit_list[1].alive
            and action_timer > ACTION_COOLDOWN
        ):
            # Bandit 2 action
            if bandit_list[1].should_heal and random.random() < BANDIT_HEAL_CHANCE:
                healed = bandit_list[1].use_potion()
                damage_text_group.add(
                    DamageText(
                        bandit_list[1].rect.centerx,
                        bandit_list[1].rect.y,
                        str(healed),
                        GREEN,
                    )
                )

            else:
                dmg = bandit_list[1].attack(player)
                damage_text_group.add(
                    DamageText(player.rect.centerx, player.rect.y, str(dmg), RED)
                )

            current_fighter = 0
            action_timer = 0

        # Draw fighters
        player.draw(window)
        for bandit in bandit_list:
            bandit.draw(window)

        # Draw damage
        damage_text_group.draw(window)

        # Check game over
        if game_state == GameState.run:
            if not player.alive:
                game_state = GameState.player_loss

            elif all([not bandit.alive for bandit in bandit_list]):
                game_state = GameState.player_win
        else:
            if restart_button.draw():
                main()

        if game_state == GameState.player_win:
            window.blit(assets["victory"], (250, 50))
        elif game_state == GameState.player_loss:
            window.blit(assets["defeat"], (290, 50))

        pygame.display.update()

    # Exit pygame
    pygame.quit()


if __name__ == "__main__":
    main()

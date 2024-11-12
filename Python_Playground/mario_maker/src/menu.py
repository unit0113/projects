import pygame
from pygame.image import load
from settings import *


class Menu:
    def __init__(self):
        self.display_surface = pygame.display.get_surface()
        self.create_data()
        self.create_buttons()

    def create_data(self):
        self.menu_surfs = {}
        for key, value in EDITOR_DATA.items():
            if value["menu"]:
                if not value["menu"] in self.menu_surfs:
                    self.menu_surfs[value["menu"]] = [(key, load(value["menu_surf"]))]
                else:
                    self.menu_surfs[value["menu"]].append(
                        (key, load(value["menu_surf"]))
                    )

    def create_buttons(self):
        # Menu area
        size = 180
        margin = 6
        self.rect = pygame.Rect(
            WINDOW_WIDTH - size - margin, WINDOW_HEIGHT - size - margin, size, size
        )

        # Buttons
        generic_button_rect = pygame.Rect(
            self.rect.topleft, (self.rect.width / 2, self.rect.height / 2)
        )
        button_margin = 5
        self.tile_button_rect = generic_button_rect.copy().inflate(
            -button_margin, -button_margin
        )

        self.coin_button_rect = (
            generic_button_rect.copy()
            .move(self.rect.width / 2, 0)
            .inflate(-button_margin, -button_margin)
        )

        self.enemy_button_rect = (
            generic_button_rect.copy()
            .move(self.rect.width / 2, self.rect.height / 2)
            .inflate(-button_margin, -button_margin)
        )

        self.palm_button_rect = (
            generic_button_rect.copy()
            .move(0, self.rect.height / 2)
            .inflate(-button_margin, -button_margin)
        )

        self.buttons = pygame.sprite.Group()
        Button(self.tile_button_rect, self.buttons, self.menu_surfs["terrain"])
        Button(self.coin_button_rect, self.buttons, self.menu_surfs["coin"])
        Button(self.enemy_button_rect, self.buttons, self.menu_surfs["enemy"])
        Button(
            self.palm_button_rect,
            self.buttons,
            self.menu_surfs["palm fg"],
            self.menu_surfs["palm bg"],
        )

    def click(self, mouse_pos, mouse_button):
        for sprite in self.buttons:
            if sprite.rect.collidepoint(mouse_pos):
                if mouse_button[1] and sprite.items["alt"]:  # Middle click
                    sprite.main_active = not sprite.main_active
                elif mouse_button[2]:  # Right click
                    sprite.switch()

                return sprite.get_id()

    def highlight_indicator(self, index):
        if EDITOR_DATA[index]["menu"] == "terrain":
            pygame.draw.rect(
                self.display_surface,
                BUTTON_LINE_COLOR,
                self.tile_button_rect.inflate(4, 4),
                5,
                4,
            )
        elif EDITOR_DATA[index]["menu"] == "coin":
            pygame.draw.rect(
                self.display_surface,
                BUTTON_LINE_COLOR,
                self.coin_button_rect.inflate(4, 4),
                5,
                4,
            )
        elif EDITOR_DATA[index]["menu"] == "enemy":
            pygame.draw.rect(
                self.display_surface,
                BUTTON_LINE_COLOR,
                self.enemy_button_rect.inflate(4, 4),
                5,
                4,
            )
        elif EDITOR_DATA[index]["menu"] in ("palm fg", "palm bg"):
            pygame.draw.rect(
                self.display_surface,
                BUTTON_LINE_COLOR,
                self.palm_button_rect.inflate(4, 4),
                5,
                4,
            )

    def display(self, index):
        self.buttons.update()
        self.buttons.draw(self.display_surface)
        self.highlight_indicator(index)


class Button(pygame.sprite.Sprite):
    def __init__(self, rect, group, items, items_alt=None) -> None:
        super().__init__(group)
        self.image = pygame.Surface(rect.size)
        self.rect = rect

        # Items
        self.items = {"main": items, "alt": items_alt}
        self.index = 0
        self.main_active = True

    def update(self):
        self.image.fill(BUTTON_BG_COLOR)
        surf = self.items["main" if self.main_active else "alt"][self.index][1]
        rect = surf.get_rect(center=(self.rect.width / 2, self.rect.height / 2))
        self.image.blit(surf, rect)

    def get_id(self):
        return self.items["main" if self.main_active else "alt"][self.index][0]

    def switch(self):
        self.index += 1
        self.index = (
            0
            if self.index >= len(self.items["main" if self.main_active else "alt"])
            else self.index
        )

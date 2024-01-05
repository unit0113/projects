import pygame
from random import choice

from .state import State
from .player_ship_data import PLAYER_SHIP_DATA
from .functions import (
    Text,
    create_mixed_stacked_text,
    draw_lines,
)
from .settings import WIDTH, HEIGHT, FRAME_TIME, KEY_PRESS_DELAY, GREY, MAGENTA
from .button import Button
from .status_bar import StatusBar
from .state_run import RunState

Y_GAP = 40
MAX_WALK = 5


class ShipSelectState(State):
    def __init__(self, game) -> None:
        super().__init__(game)

        self.key_down = False
        self.key_timer = 0
        self.selected_ship = 0
        self.ships = list(PLAYER_SHIP_DATA.keys())

        # Menu text
        characteristics = [
            "Base Stats",
            "  Health",
            "  Speed",
            "Shield",
            "  Strength",
            "  Cooldown",
            "  Regen",
            "Firepower",
            "  Starting",
            "  Potential",
        ]
        text_fonts = [
            self.game.assets["small_font"]
            if item[0] == " "
            else self.game.assets["med_font"]
            for item in characteristics
        ]
        text_colors = [GREY if item[0] == " " else MAGENTA for item in characteristics]
        self.characteristics = create_mixed_stacked_text(
            characteristics, WIDTH // 2 - 275, 200, Y_GAP, text_fonts, text_colors, True
        )
        self._get_ship_min_max_data()

        # Menu outline
        self.points1 = [
            (WIDTH // 2 - 150, 100),
            (WIDTH // 2 + 150, 100),
            (WIDTH // 2 + 150, 150),
            (WIDTH // 2 - 150, 150),
            (WIDTH // 2 - 150, 100),
        ]
        self.ship_title_position = (WIDTH // 2, 125)

        self.points2 = [
            (WIDTH // 2 - 150, 125),
            (WIDTH // 2 - 300, 125),
            (WIDTH // 2 - 300, HEIGHT - 300),
            (WIDTH // 2 + 300, HEIGHT - 300),
            (WIDTH // 2 + 300, 125),
            (WIDTH // 2 + 150, 125),
        ]

        # Semi-transparent backround
        # Half height due to overlap with main rect
        self.title_backround_rect = pygame.Surface((300, 25))
        self.title_backround_rect.set_alpha(128)
        self.title_backround_rect.fill((0, 0, 0))

        self.main_backround_rect = pygame.Surface((600, HEIGHT - 425))
        self.main_backround_rect.set_alpha(128)
        self.main_backround_rect.fill((0, 0, 0))

        # Status bars
        self.health_status_bar = StatusBar(
            (WIDTH // 2 + 50, 200 + Y_GAP),
            (225, 25),
            self.min_hp,
            self.max_hp,
            PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["multipliers"]["hp"],
        )
        self.speed_status_bar = StatusBar(
            (WIDTH // 2 + 50, 200 + 2 * Y_GAP),
            (225, 25),
            self.min_speed,
            self.max_speed,
            PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["multipliers"]["speed"],
        )
        self.shield_strength_status_bar = StatusBar(
            (WIDTH // 2 + 50, 200 + 4 * Y_GAP),
            (225, 25),
            self.min_shield_strength,
            self.max_shield_strength,
            PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["multipliers"][
                "shield_strength"
            ],
        )
        self.shield_cooldown_status_bar = StatusBar(
            (WIDTH // 2 + 50, 200 + 5 * Y_GAP),
            (225, 25),
            self.min_shield_cooldown,
            self.max_shield_cooldown,
            PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["multipliers"][
                "shield_cooldown"
            ],
        )
        self.shield_regen_status_bar = StatusBar(
            (WIDTH // 2 + 50, 200 + 6 * Y_GAP),
            (225, 25),
            self.min_shield_regen,
            self.max_shield_regen,
            PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["multipliers"][
                "shield_regen"
            ],
        )
        self.starting_firepower_status_bar = StatusBar(
            (WIDTH // 2 + 50, 200 + 8 * Y_GAP),
            (225, 25),
            self.min_starting_firepower,
            self.max_starting_firepower,
            self._calculate_starting_firepower(
                PLAYER_SHIP_DATA[self.ships[self.selected_ship]]
            ),
        )
        self.potential_firepower_status_bar = StatusBar(
            (WIDTH // 2 + 50, 200 + 9 * Y_GAP),
            (225, 25),
            self.min_potential_firepower,
            self.max_potential_firepower,
            self._calculate_potential_firepower(
                PLAYER_SHIP_DATA[self.ships[self.selected_ship]]
            ),
        )

        # Ship animation
        self.ship_sprites = {}
        self._swap_ship()
        self.frame_index = 0
        self.last_frame = pygame.time.get_ticks()
        self._reset_image()
        self.ship_pos = [WIDTH // 2 - self.ship_image.get_width() // 2, HEIGHT - 250]
        self.ship_pos_start = self.ship_pos

        # Ship select arrows
        self.left_arrow_normal_image = pygame.transform.scale_by(
            self.game.assets["left_arrow"], 0.25
        )
        self.left_arrow_image = self.left_arrow_normal_image
        self.left_arrow_selected_image = pygame.transform.scale_by(
            self.game.assets["left_arrow_filled"], 0.25
        )
        self.left_arrow_rect = self.left_arrow_image.get_rect()
        self.left_arrow_rect.midright = (
            WIDTH // 2 - self.ship_image.get_width() // 2,
            HEIGHT - 250 + self.ship_image.get_height() // 2,
        )

        self.right_arrow_normal_image = pygame.transform.scale_by(
            self.game.assets["right_arrow"], 0.25
        )
        self.right_arrow_image = self.right_arrow_normal_image
        self.right_arrow_selected_image = pygame.transform.scale_by(
            self.game.assets["right_arrow_filled"], 0.25
        )
        self.right_arrow_rect = self.right_arrow_image.get_rect()
        self.right_arrow_rect.midleft = (
            WIDTH // 2 + self.ship_image.get_width() // 2,
            HEIGHT - 250 + self.ship_image.get_height() // 2,
        )

        # Select ship button
        self.select_button = Button(
            WIDTH // 2, HEIGHT - 400, "Select Ship", self.game.assets["font"], 310
        )

    def _get_weapon_name(self) -> str:
        """Parses the weapon name for display

        Returns:
            str: Weapon name
        """
        if PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["primary_weapons"]:
            return PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["primary_weapons"][
                0
            ][0].capitalize()
        # If ship is secondary weapon only
        else:
            return (
                PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["secondary_weapons"][0]
                .replace("_", " ")
                .capitalize()
            )

    def _swap_ship(self) -> None:
        """Generates the title text and loads ship sprites"""
        self.title_text = Text(
            self.ships[self.selected_ship], (WIDTH // 2, 125), self.game.assets["font"]
        )
        weapon = self._get_weapon_name()
        self.weapon_text = Text(
            weapon,
            (WIDTH // 2 + 50, 200 + 7 * Y_GAP),
            self.game.assets["small_font"],
            True,
        )

        # Get level sprite sheet, memoized
        if self.selected_ship not in self.ship_sprites.keys():
            sprite_sheet = self.game.assets["sprite_sheets"][
                PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["sprite_sheet"]
            ]
            size_h = sprite_sheet.get_width() // 4
            size_v = sprite_sheet.get_height() // 5

            sprites = []
            for column in range(4):
                sprite = sprite_sheet.subsurface(column * size_h, 0, size_h, size_v)
                sprite = pygame.transform.scale_by(sprite, 2)
                sprites.append(sprite)
            self.ship_sprites[self.selected_ship] = sprites

        # Update status bars
        self.health_status_bar.update_value(
            PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["multipliers"]["hp"]
        )
        self.speed_status_bar.update_value(
            PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["multipliers"]["speed"]
        )
        self.shield_strength_status_bar.update_value(
            PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["multipliers"][
                "shield_strength"
            ]
        )
        self.shield_cooldown_status_bar.update_value(
            PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["multipliers"][
                "shield_cooldown"
            ]
        )
        self.shield_regen_status_bar.update_value(
            PLAYER_SHIP_DATA[self.ships[self.selected_ship]]["multipliers"][
                "shield_regen"
            ]
        )
        self.starting_firepower_status_bar.update_value(
            self._calculate_starting_firepower(
                PLAYER_SHIP_DATA[self.ships[self.selected_ship]]
            )
        )
        self.potential_firepower_status_bar.update_value(
            self._calculate_potential_firepower(
                PLAYER_SHIP_DATA[self.ships[self.selected_ship]]
            )
        )

    def _get_ship_min_max_data(self) -> None:
        """Parses the player ship data and finds the min and max values of characteristics"""
        self.min_hp = 100
        self.max_hp = 0
        self.min_speed = 100
        self.max_speed = 0
        self.min_shield_strength = 100
        self.max_shield_strength = 0
        self.min_shield_cooldown = 100
        self.max_shield_cooldown = 0
        self.min_shield_regen = 100
        self.max_shield_regen = 0
        self.min_starting_firepower = 100
        self.max_starting_firepower = 0
        self.min_potential_firepower = 100
        self.max_potential_firepower = 0

        # Find max and min of ship statistics
        for ship in PLAYER_SHIP_DATA:
            # HP
            if PLAYER_SHIP_DATA[ship]["multipliers"]["hp"] < self.min_hp:
                self.min_hp = PLAYER_SHIP_DATA[ship]["multipliers"]["hp"]
            if PLAYER_SHIP_DATA[ship]["multipliers"]["hp"] > self.max_hp:
                self.max_hp = PLAYER_SHIP_DATA[ship]["multipliers"]["hp"]
            # Speed
            if PLAYER_SHIP_DATA[ship]["multipliers"]["speed"] < self.min_speed:
                self.min_speed = PLAYER_SHIP_DATA[ship]["multipliers"]["speed"]
            if PLAYER_SHIP_DATA[ship]["multipliers"]["speed"] > self.max_speed:
                self.max_speed = PLAYER_SHIP_DATA[ship]["multipliers"]["speed"]
            # Shield strength
            if (
                PLAYER_SHIP_DATA[ship]["multipliers"]["shield_strength"]
                < self.min_shield_strength
            ):
                self.min_shield_strength = PLAYER_SHIP_DATA[ship]["multipliers"][
                    "shield_strength"
                ]
            if (
                PLAYER_SHIP_DATA[ship]["multipliers"]["shield_strength"]
                > self.max_shield_strength
            ):
                self.max_shield_strength = PLAYER_SHIP_DATA[ship]["multipliers"][
                    "shield_strength"
                ]
            # Shield cooldown
            if (
                PLAYER_SHIP_DATA[ship]["multipliers"]["shield_cooldown"]
                < self.min_shield_cooldown
            ):
                self.min_shield_cooldown = PLAYER_SHIP_DATA[ship]["multipliers"][
                    "shield_cooldown"
                ]
            if (
                PLAYER_SHIP_DATA[ship]["multipliers"]["shield_cooldown"]
                > self.max_shield_cooldown
            ):
                self.max_shield_cooldown = PLAYER_SHIP_DATA[ship]["multipliers"][
                    "shield_cooldown"
                ]
            # Shield cooldown
            if (
                PLAYER_SHIP_DATA[ship]["multipliers"]["shield_regen"]
                < self.min_shield_regen
            ):
                self.min_shield_regen = PLAYER_SHIP_DATA[ship]["multipliers"][
                    "shield_regen"
                ]
            if (
                PLAYER_SHIP_DATA[ship]["multipliers"]["shield_regen"]
                > self.max_shield_regen
            ):
                self.max_shield_regen = PLAYER_SHIP_DATA[ship]["multipliers"][
                    "shield_regen"
                ]
            # Firepower
            starting_firepower = self._calculate_starting_firepower(
                PLAYER_SHIP_DATA[ship]
            )
            if starting_firepower < self.min_starting_firepower:
                self.min_starting_firepower = starting_firepower
            if starting_firepower > self.max_starting_firepower:
                self.max_starting_firepower = starting_firepower

            potential_firepower = self._calculate_potential_firepower(
                PLAYER_SHIP_DATA[ship]
            )
            if potential_firepower < self.min_potential_firepower:
                self.min_potential_firepower = potential_firepower
            if potential_firepower > self.max_potential_firepower:
                self.max_potential_firepower = potential_firepower

    def _calculate_starting_firepower(self, ship_data: dict) -> float:
        """Determines the starting firepower of the provided ship

        Args:
            ship_data (dict): Ship data to calculate from

        Returns:
            float: Starting firepower of provided ship
        """
        return 2 * len(ship_data["primary_weapons"]) + len(
            ship_data["secondary_weapons"]
        )

    def _calculate_potential_firepower(self, ship_data: dict) -> float:
        """Determines the potential firepower of the provided ship

        Args:
            ship_data (dict): Ship data to calculate from

        Returns:
            float: Potential firepower of provided ship
        """
        return 2 * len(ship_data["primary_weapons"]) + len(
            ship_data["secondary_offsets"]
        )

    def update(self, dt: float, **kwargs) -> None:
        """Update game object in game loop

        Args:
            dt (float): time since last frame
        """
        # Mouse over arrows
        pos = pygame.mouse.get_pos()
        if self.left_arrow_rect.collidepoint(pos):
            self.left_arrow_image = self.left_arrow_selected_image
        else:
            self.left_arrow_image = self.left_arrow_normal_image
        if self.right_arrow_rect.collidepoint(pos):
            self.right_arrow_image = self.right_arrow_selected_image
        else:
            self.right_arrow_image = self.right_arrow_normal_image

        # Mouse over select button
        self.select_button.set_highlight(self.select_button.mouse_over(pos))

        # Prevent rapid changing of menu selection
        if self.key_down:
            self.key_timer += dt
            if self.key_timer > KEY_PRESS_DELAY:
                self.key_down = False
                self.key_timer = 0
            return

        keys = pygame.key.get_pressed()
        # Change menu selection
        if keys[pygame.K_LEFT]:
            self._swap_left()
        elif keys[pygame.K_RIGHT]:
            self._swap_right()
        self._animate()

    def _swap_left(self) -> None:
        """Select the ship to the left"""
        self.key_down = True
        self.selected_ship -= 1
        if self.selected_ship < 0:
            self.selected_ship = len(self.ships) - 1
        self._swap_ship()

    def _swap_right(self) -> None:
        """Select the ship to the right"""
        self.key_down = True
        self.selected_ship += 1
        if self.selected_ship >= len(self.ships):
            self.selected_ship = 0
        self._swap_ship()

    def _animate(self) -> None:
        """Animate ship sprite"""
        if pygame.time.get_ticks() > self.last_frame + FRAME_TIME:
            # Reset frame counter and increment frame
            self.last_frame = pygame.time.get_ticks()
            self.frame_index += 1
            if self.frame_index >= len(self.ship_sprites[self.selected_ship]):
                # Loop animation
                self.frame_index = 0
            self._reset_image()
            self._random_walk()

    def _reset_image(self) -> None:
        """Loads correct image based on animation"""
        self.ship_image = self.ship_sprites[self.selected_ship][self.frame_index]

    def _random_walk(self) -> None:
        """Randomly moves ship sprite around"""
        x_move = choice([-1, 0, 1])
        y_move = choice([-1, 0, 1])
        self.ship_pos[0] = max(
            self.ship_pos_start[0] - MAX_WALK,
            min(self.ship_pos_start[0] + MAX_WALK, self.ship_pos[0] + x_move),
        )
        self.ship_pos[1] = max(
            self.ship_pos_start[1] - MAX_WALK,
            min(self.ship_pos_start[1] + MAX_WALK, self.ship_pos[1] + y_move),
        )

    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """
        # Draw transparent rects
        window.blit(self.title_backround_rect, self.points1[0])
        window.blit(self.main_backround_rect, self.points2[1])

        # Draw outline
        draw_lines(window, self.points1, 4)
        draw_lines(window, self.points2, 4)

        # Draw text
        self.title_text.draw(window)
        for item in self.characteristics:
            item.draw(window)
        self.weapon_text.draw(window)

        # Draw ship
        window.blit(self.ship_image, self.ship_pos)

        # Draw arrows
        window.blit(self.left_arrow_image, self.left_arrow_rect)
        window.blit(self.right_arrow_image, self.right_arrow_rect)

        # Draw button
        self.select_button.draw(window)

        # Draw status bars
        self.health_status_bar.draw(window)
        self.speed_status_bar.draw(window)
        self.shield_strength_status_bar.draw(window)
        self.shield_cooldown_status_bar.draw(window)
        self.shield_regen_status_bar.draw(window)
        self.starting_firepower_status_bar.draw(window)
        self.potential_firepower_status_bar.draw(window)

    def enter(self, **kwargs) -> None:
        """Actions to perform upon entering the state"""
        pass

    def exit(self) -> None:
        """Actions to perform upon exiting the state"""

        self.game.set_player(self.ships[self.selected_ship])
        self.next_state = RunState(self.game)

    def process_events(self, events: list[pygame.event.Event]):
        """Handle game events

        Args:
            events (list[pygame.event.Event]): events to handle
        """

        # Select option mouse
        for event in events:
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                if self.left_arrow_rect.collidepoint(pos):
                    self._swap_left()
                if self.right_arrow_rect.collidepoint(pos):
                    self._swap_right()
                if self.select_button.mouse_over(pos):
                    self.should_exit = True

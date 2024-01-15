import pygame
import pytweening as tween

from .state import State
from .functions import draw_lines, random_walk
from .text import Text
from .status_bar import StatusBar
from .button import Button
from .settings import WIDTH, HEIGHT, BLACK, FRAME_TIME

FADE_OUT_TIME = 1
STAGE_2_START_TIME = 2
STAGE_3_START_TIME = 2.25
STAGE_4_START_TIME = 3.75
STAGE_5_START_TIME = 4
STAGE_6_START_TIME = 5
STAGE_7_START_TIME = 5.5
END_MANUEVER_TIME = 7
MANEUVER_DIST = WIDTH / 4


class ShipSelectRunTransitionState(State):
    def __init__(
        self,
        game,
        characteristics: list[Text],
        points1: list[tuple[int, int]],
        points2: list[tuple[int, int]],
        title_backround_rect: pygame.Surface,
        main_backround_rect: pygame.Surface,
        status_bars: list[StatusBar],
        select_button: Button,
        title_text: Text,
        weapon_text: Text,
        sprite_sheet: pygame.Surface,
        ship_pos: list[int, int],
    ) -> None:
        super().__init__(game)
        self.game = game
        self.characteristics = characteristics
        self.points1 = points1
        self.points2 = points2
        self.title_backround_rect = title_backround_rect
        self.main_backround_rect = main_backround_rect
        self.status_bars = status_bars
        self.select_button = select_button
        self.title_text = title_text
        self.weapon_text = weapon_text

        # Parse sprite sheet
        self.sprites = self.game.player.split_sprite_sheet(sprite_sheet, 5, 4, scale=2)
        self.ship_pos = ship_pos

        self.black_screen_surf = pygame.Surface((WIDTH, HEIGHT))
        self.black_screen_surf.fill(BLACK)
        self.black_screen_surf.set_alpha(0)

        # Tween
        self.timer = 0
        self.start_x = self.ship_pos[0]

        # Ship animation
        self.last_frame = pygame.time.get_ticks()
        self.frame_index = 0
        self.last_frame_index = 0
        self.orientation = "level"
        self.ship_image = self.sprites[self.orientation][self.frame_index]

    def update(self, dt: float, **kwargs) -> None:
        """Update game object in game loop

        Args:
            dt (float): time since last frame
        """

        self.timer += dt
        self.last_frame_index = self.frame_index

        # Continue animating ship
        if pygame.time.get_ticks() > self.last_frame + FRAME_TIME:
            self.ship_pos = random_walk(self.ship_pos, self.ship_pos)
            self._update_image()

        # Fade out ship select objects
        if self.timer < FADE_OUT_TIME:
            # Fade out ship select objects
            self.black_screen_surf.set_alpha(
                255 * tween.easeInOutSine(self.timer / FADE_OUT_TIME)
            )

        # Manuever ship
        elif FADE_OUT_TIME <= self.timer < STAGE_7_START_TIME:
            # Set background to black
            if self.black_screen_surf.get_alpha() != 255:
                self.black_screen_surf.set_alpha(255)
            # Stage 1
            if self.timer < STAGE_2_START_TIME:
                self.orientation = "left"
                self.ship_pos[0] = self.start_x - MANEUVER_DIST * tween.easeInOutSine(
                    (self.timer - FADE_OUT_TIME) / (STAGE_2_START_TIME - FADE_OUT_TIME)
                )
            # Stage 2
            elif self.timer < STAGE_3_START_TIME:
                self.orientation = "level"
                self.start_x = self.ship_pos[0]
            # Stage 3
            elif self.timer < STAGE_4_START_TIME:
                self.orientation = "right"
                self.ship_pos[
                    0
                ] = self.start_x + 2 * MANEUVER_DIST * tween.easeInOutSine(
                    (self.timer - STAGE_3_START_TIME)
                    / (STAGE_4_START_TIME - STAGE_3_START_TIME)
                )
            # Stage 4
            elif self.timer < STAGE_5_START_TIME:
                self.orientation = "level"
                self.start_x = self.ship_pos[0]
            # Stage 5
            elif self.timer < STAGE_6_START_TIME:
                self.orientation = "left"
                self.ship_pos[0] = self.start_x - MANEUVER_DIST * tween.easeInOutSine(
                    (self.timer - STAGE_5_START_TIME)
                    / (STAGE_6_START_TIME - STAGE_5_START_TIME)
                )
            # Stage 6
            else:
                self.orientation = "level"
                self.start_y = self.ship_pos[1]
        # Stage 7
        elif self.timer < END_MANUEVER_TIME:
            self.orientation = "level"
            self.ship_pos[1] = self.start_y - (self.start_y + 400) * tween.easeInQuad(
                (self.timer - STAGE_7_START_TIME)
                / (END_MANUEVER_TIME - STAGE_7_START_TIME)
            )

        # Update sprite image
        if self.frame_index != self.last_frame_index:
            self.ship_image = self.sprites[self.orientation][self.frame_index]

    def _update_image(self) -> None:
        self.last_frame = pygame.time.get_ticks()
        self.frame_index += 1
        if self.frame_index >= len(self.sprites[self.orientation]):
            # Loop animation
            self.frame_index = 0

    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """
        if self.timer < FADE_OUT_TIME:
            # Draw transparent rects
            window.blit(self.title_backround_rect, self.points1[0])
            window.blit(self.main_backround_rect, self.points2[1])

            # Draw outline
            draw_lines(window, self.points1, 4)
            draw_lines(window, self.points2, 4)

            # Draw text
            self.title_text.draw(window)
            self.weapon_text.draw(window)
            for item in self.characteristics:
                item.draw(window)

            # Draw status bars
            for status_bar in self.status_bars:
                status_bar.draw(window)

            # Draw button
            self.select_button.draw(window)

        # Draw black screen
        window.blit(self.black_screen_surf, (0, 0))

        # Draw ship
        window.blit(self.ship_image, self.ship_pos)

    def enter(self, **kwargs) -> None:
        """Actions to perform upon entering the state"""
        pass

    def exit(self) -> None:
        """Actions to perform upon exiting the state"""

        pass

    def process_events(self, events: list[pygame.event.Event]):
        """Handle game events

        Args:
            events (list[pygame.event.Event]): events to handle
        """

        pass

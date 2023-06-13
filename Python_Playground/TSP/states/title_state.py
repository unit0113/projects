import pygame
import pytweening as tween

from states.state import State
from src.settings import HEIGHT, WIDTH
from src.colors import BLACK, MENU_PURPLE

TITLE_HEIGHT_POS_START = -100
TITLE_HEIGHT_POS_END = 50
TITLE_OUTER_REC_SIZE = 20
TITLE_INNER_REC_SIZE = 14

TITLE_TWEEN_DURATION = 2
MAP_DELAY = 1
MAP_TWEEN_DURATION = 3
BUTTON_DELAY = 0.25

MAP_X = WIDTH - 900
MAP_Y = HEIGHT - 900

LOAD_TIME = 10


class TitleState(State):
    def __init__(self, game) -> None:
        self.game = game
        self.timer = 0

        # Title Card
        self.title_font = pygame.font.SysFont('verdana', 40, bold=True)
        self.title_text = self.title_font.render('Traveling Salesman Problem Approximation', 1, MENU_PURPLE) 

        self.start_x = WIDTH // 2 - self.title_text.get_width() // 2
        self.start_y = TITLE_HEIGHT_POS_START + self.title_text.get_height() // 2
        self.offset = 0

        # Title outlines
        self.title_range = TITLE_HEIGHT_POS_END - TITLE_HEIGHT_POS_START
        self.title_rect_outer = pygame.Rect(self.start_x - TITLE_OUTER_REC_SIZE // 2, self.start_y - TITLE_OUTER_REC_SIZE // 2, self.title_text.get_width() + TITLE_OUTER_REC_SIZE, self.title_text.get_height() + TITLE_OUTER_REC_SIZE)
        self.title_rect_inner = pygame.Rect(self.start_x - TITLE_INNER_REC_SIZE // 2, self.start_y - TITLE_INNER_REC_SIZE // 2, self.title_text.get_width() + TITLE_INNER_REC_SIZE, self.title_text.get_height() + TITLE_INNER_REC_SIZE)

        # Initiate game button tweening
        for index, button in enumerate(self.game.buttons):
            button.set_tween(tween.easeOutQuad, 1.5 + index * BUTTON_DELAY, 2, True, 400)   # Issue with buttons not traversing to finish point

    def update(self, dt: float, actions: list) -> None:
        # Update buttons:
        for button in self.game.buttons:
            button.update(self.timer)

        # Tween map, fade in
        if MAP_DELAY < self.timer < MAP_DELAY + MAP_TWEEN_DURATION:
            self.game.assets_dict['map'].set_alpha(255 * tween.easeInOutQuad((self.timer - MAP_DELAY) / (MAP_DELAY + MAP_TWEEN_DURATION)))

        # Tween Title Card in from off screen
        if self.timer < TITLE_TWEEN_DURATION:
            self.offset = self.title_range * tween.easeInOutQuad(self.timer / TITLE_TWEEN_DURATION)
            self.title_rect_outer.y = self.start_y - TITLE_OUTER_REC_SIZE // 2 + self.offset
            self.title_rect_inner.y = self.start_y - TITLE_INNER_REC_SIZE // 2 + self.offset

        # Transfer control to menu state
        if self.timer > LOAD_TIME:
            self.game.set_state('main_menu')

        self.timer += dt

    def draw(self) -> None:
        self.game.window.fill(BLACK)

        # Buttons
        for button in self.game.buttons:
            button.draw(self.game.window)

        # Map
        self.game.window.blit(self.game.assets_dict['map'], (MAP_X, MAP_Y))

        # Title bar
        pygame.draw.rect(self.game.window, MENU_PURPLE, self.title_rect_outer, border_radius=10)
        pygame.draw.rect(self.game.window, BLACK, self.title_rect_inner, border_radius=10)
        self.game.window.blit(self.title_text, (self.start_x, self.start_y + self.offset))

        pygame.display.update()
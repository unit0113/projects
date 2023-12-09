import pygame

from .state import State
from .player_ship import PlayerShip


class TestState(State):
    def __init__(self, game) -> None:
        super().__init__(game)

    def update(self, dt: float, **kwargs) -> None:
        self.game.update_player(dt)
        self.game.update_projectiles(dt)

        self.game.fire()
        self.game.remove_offscreen_objects()

    def draw(self, window: pygame.Surface) -> None:
        self.game.draw_projectiles(window)
        self.game.draw_player(window)

    def enter(self, **kwargs) -> None:
        self.game.set_player(PlayerShip("boomerang"))

    def exit(self) -> None:
        pass

    def process_event(self, event: pygame.event.Event):
        pass

import pygame

from .state import State
from .player_ship import PlayerShip


class TestState(State):
    def __init__(self, game) -> None:
        super().__init__(game)

    def update(self, dt: float, **kwargs) -> None:
        """Update game object in game loop

        Args:
            dt (float): time since last frame
        """

        self.game.check_collisions()
        self.game.update_player(dt)
        self.game.update_enemies(dt)
        self.game.update_projectiles(dt)

        self.game.fire()
        self.game.remove_offscreen_objects()
        self.game.spawn_enemies()

    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        self.game.draw_projectiles(window)
        self.game.draw_enemies(window)
        self.game.draw_player(window)
        self.game.draw_UI(window)

    def enter(self, **kwargs) -> None:
        """Actions to perform upon entering the state"""
        self.game.set_player(PlayerShip("Test"))

    def exit(self) -> None:
        """Actions to perform upon exiting the state"""

        pass

    def process_events(self, events: list[pygame.event.Event]):
        """Handle game events

        Args:
            events (list[pygame.event.Event]): events to handle
        """

        pass

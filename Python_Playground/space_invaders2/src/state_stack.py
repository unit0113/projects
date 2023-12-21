# https://github.com/steelx/py-rpg-01/tree/main
import pygame
from typing import Optional

from .state import State


class StateStack:
    def __init__(self):
        self.states: list[State] = []

    def push(self, state: State) -> None:
        """Add new state onto the stack and enter. Exit previous top state

        Args:
            state (State): _description_
        """

        if self.states:
            self.states[-1].exit()
        self.states.append(state)
        state.enter()

    def pop(self) -> Optional[State]:
        """Remove top state from stack and exit. If state leads to new state, push that state onto the stack

        Returns:
            Optional[State]: state that was popped
        """

        if self.states:
            top_state = self.states.pop()
            top_state.exit()
            next_state = top_state.next_state
            if next_state:
                self.push(next_state)
            return top_state
        return None

    def top(self) -> Optional[State]:
        """Peek at top state on stack

        Returns:
            Optional[State]: top state on stack
        """

        if self.states:
            return self.states[-1]
        return None

    def update(self, dt: float):
        """Update game objects in game loop

        Args:
            dt (float): time since last frame
        """

        if self.states:
            self.states[-1].update(dt)
            if self.states[-1].should_exit:
                self.pop()

    def draw(self, window: pygame.Surface):
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        for state in self.states:
            state.draw(window)

    def process_event(self, event: pygame.event.Event):
        """Handle specific event

        Args:
            event (pygame.event.Event): event to handle
        """

        if self.states:
            self.states[-1].process_event(event)

    def is_empty(self) -> bool:
        """Check whether state stack is empty

        Returns:
            bool: stack is empty
        """

        return len(self.states) == 0

    def empty(self) -> None:
        """Empties the current stack"""

        self.states.clear()

    def __str__(self):
        return "\n".join(str(state) for state in self.states)

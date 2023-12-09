import pygame
from typing import Optional

from .state import State


class StateStack:
    def __init__(self):
        self.states: list[State] = []

    def push(self, state: State):
        if self.states:
            self.states[-1].exit()
        self.states.append(state)
        state.enter()

    def pop(self) -> Optional[State]:
        if self.states:
            top_state = self.states.pop()
            top_state.exit()
            next_state = top_state.next_state
            if next_state:
                self.push(next_state)
            return top_state
        return None

    def top(self) -> Optional[State]:
        if self.states:
            return self.states[-1]
        return None

    def update(self, dt: float):
        if self.states:
            self.states[-1].update(dt)
            if self.states[-1].should_exit:
                self.pop()

    def draw(self, window: pygame.Surface):
        for state in self.states:
            state.draw(window)

    def process_event(self, event: pygame.event.Event):
        if self.states:
            self.states[-1].process_event(event)

    def is_empty(self) -> bool:
        return len(self.states) == 0

    def __str__(self):
        return "\n".join(str(state) for state in self.states)

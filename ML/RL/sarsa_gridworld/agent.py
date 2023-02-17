import copy
import numpy as np

class Agent(object):
    def __init__(self, init_y: int, init_x:int) -> None:
        self.pos = [init_y, init_x]
        self.reward = 0

    def select_e_greedily(self, Qmat: list, e:float = 0.2) -> str:
        y, x = self.pos
        choices = Qmat[y][x]
        if np.random.uniform(0, 1) < e:
            # Select random move
            return np.random.choice(list(choices.keys()))
        else:
            # Select highest Q move
            return max(choices, key=choices.get)

    def move(self, dir: str, walls: list, size: int) -> list:
        new_pos = copy.copy(self.pos)

        if dir == 'north':
            new_pos[0] += -1
        elif dir == 'south':
            new_pos[0] += 1
        elif dir == 'east':
            new_pos[1] += 1
        elif dir == 'west':
            new_pos[1] += -1

        # Check if bumped against a wall or out-of bounds
        if not self._out_of_bounds(new_pos, walls, size):
            self.pos = new_pos

        return self.pos
    
    def _out_of_bounds(self, position, walls: list, size: int) -> bool:
        return (position in walls
                or position[0] < 0
                or position[0] >= size
                or position[1] < 0
                or position[1] >= size)
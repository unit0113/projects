with open(r'AoC\2021\Day_15\input.txt', 'r') as file:
    risks = [[int(risk) for risk in list(line.strip())] for line in file.readlines()]
    
HEIGHT = len(risks)
WIDTH = len(risks[0])

class Node:
    def __init__(self, x, y, g):
        self.x = x
        self.y = y
        self.g = g
        self.h = HEIGHT - 1 - x + WIDTH - 1 - y
        self.f = self.g + self.h

    
    def recalc_f(self):
        self.f = self.g + self.h


    def __lt__(self, other):
        return self.f < other.f
                

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def a_star(risks):
    DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    start = Node(0, 0, 0)
    end = Node(HEIGHT * 5 - 1, WIDTH * 5 - 1, 9)
    open = []
    open.append(start)
    closed = []
    cycles = 0

    def valid_moves(risks, start):
        moves = []
        for move in DIRECTIONS:
            x = move[0] + start.x
            y = move[1] + start.y

            if 0 <= y < HEIGHT * 5 and 0 <= x < WIDTH * 5 and (x, y) not in closed:
                repeat = y // HEIGHT + x // WIDTH
                distance = risks[y % HEIGHT][x % WIDTH] + repeat
                distance = (distance - 1) % 9 + 1
                node = Node(x, y, distance + start.g)
                moves.append(node)

        return moves


    while open:
        cycles += 1
        current = min(open)
        open.remove(current)
        closed.append((current.x, current.y))
        if current == end:
            return current.g

        for neighbor in valid_moves(risks, current):
            if (neighbor.x, neighbor.y) in closed:
                continue
            if neighbor in open:
                old_node = [node for node in open if node == neighbor][0]
                old_node.g = min(old_node.g, neighbor.g)
                old_node.recalc.f()
            
            else:
                open.append(neighbor)


        if cycles % 5000 == 0:
            print(f"Cycle: {cycles}. Processing grid [{current.x}, {current.y}]. Current cost: {current.g}")


print(a_star(risks))
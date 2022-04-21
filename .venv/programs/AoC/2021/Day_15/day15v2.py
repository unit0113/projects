with open(r'AoC\2021\Day_15\input.txt', 'r') as file:
    risks = [[int(risk) for risk in list(line.strip())] for line in file.readlines()]
    

class Node:
    def __init__(self, x, y, g):
        self.x = x
        self.y = y
        self.g = g
        self.h = len(risks[0]) - 1 - x +  len(risks) - 1 - y
        self.f = self.g + self.h


    def __lt__(self, other):
        return self.f < other.f
        

    def __le__(self, other):
        return self.f <= other.f
                

    def __gt__(self, other):
        return self.f > other.f
                

    def __ge__(self, other):
        return self.f >= other.f
                

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
                

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y


def a_star(risks):
    start = Node(0, 0, 0)
    end = Node(len(risks[0]) - 1, len(risks) - 1, risks[len(risks[0]) - 1][len(risks) - 1])
    open = []
    open.append(start)
    closed = []
    cycles = 0

    while open:
        cycles += 1
        current = min(open)
        open.remove(current)
        closed.append(current)
        if current == end:
            return current.g

        for neighbor in valid_moves(risks, current):
            if neighbor in closed:
                continue
            if neighbor in open:
                old_node = [node for node in open if node == neighbor][0]
                if old_node.g > neighbor.g:
                    open.remove(old_node)
                    open.append(neighbor)
            else:
                open.append(neighbor)

        if cycles % 5000 == 0:
            print(f"Cycle: {cycles}. Processing grid [{current.x}, {current.y}]. Current cost: {current.g}")


def valid_moves(risks, start):
    moves = []
    poss_moves = [(start.x,start.y-1),(start.x,start.y+1),(start.x-1,start.y),(start.x+1,start.y)]
    for move in poss_moves:
        x, y = move

        if 0 <= x <= len(risks[0]) - 1 and 0 <= y <= len(risks) - 1:
            node = Node(x, y, start.g + risks[x][y])
            moves.append(node)

    return moves


def enhance_risk_map(risks):
    risks_copy = risks[:]
    for i in range(1,5):
        new_section = [[x+i if x+i <= 9 else x+i-9 for x in row] for row in risks_copy]
        risks += new_section
    
    risks_copy = risks[:]
    for i in range(1, 5):
        new_section = [[x+i if x+i <= 9 else x+i-9 for x in row] for row in risks_copy]
        risks = [a+b for a,b in zip(risks, new_section)]

    risks[0][0] = 0

    return risks

risks = enhance_risk_map(risks)
print(a_star(risks))
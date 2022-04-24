with open(r'AoC\2021\Day_15\input.txt', 'r') as file:
    risks = [[int(risk) for risk in list(line.strip())] for line in file.readlines()]


def a_star(risks, multiple):
    HEIGHT = len(risks)
    WIDTH = len(risks[0])
    NEW_HEIGHT = HEIGHT * multiple
    NEW_WIDTH = WIDTH * multiple
    DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    closed = set()
    open = [(0, (0,0))]
    cycles = 0

    while open:
        cycles += 1
        current = open.pop(0)
        current_g = current[0]
        x = current[1][0]
        y = current[1][1]
        closed.add((x, y))

        if x == NEW_WIDTH - 1 and y == NEW_HEIGHT - 1:
            print(current_g)
            break

        for option in DIRECTIONS:
            new_x = x + option[0]
            new_y = y + option[1]

            if 0 <= new_x < NEW_WIDTH and 0 <= new_y < NEW_HEIGHT and (new_x, new_y) not in closed:
                grid_x = new_x % WIDTH
                grid_y = new_y % HEIGHT
                repeat = new_x // WIDTH + new_y // HEIGHT
                new_dist = risks[grid_y][grid_x] + repeat
                new_dist = (new_dist - 1) % 9 + 1
                new_g = new_dist + current_g
                open.append((new_g, (new_x, new_y)))
                closed.add((new_x, new_y))

        if cycles % 5000 == 0:
            print(f"Cycle: {cycles}. Processing grid [{current[1][0]}, {current[1][1]}]. Current cost: {current[0]}")

        open = sorted(open)


def find_next(array):
    min_cost = array[0][0]
    min_index = 0
    for index, item in enumerate(array[1:]):
        if item[0] <= min_cost:
            min_cost = item[0]
            min_index = index
    
    return min_index


a_star(risks, 5)
points = set()
commands = []
with open(r'AoC\2021\Day_13\input.txt', 'r') as file:
    for line in file.readlines():
        if line == '\n':
            continue
        if 'fold' in line:
            dir, amt = line.replace('fold along ', '').strip().split('=')
            commands.append((dir, int(amt)))
        else:
            x, y = line.strip().split(',')
            points.add((int(x), int(y)))


#points = set([(6,10), (0,14), (9,10), (0,3), (10,4), (4,11), (6,0), (6,12), (4,1), (0,13), (10,12), (3,4), (3,0), (8,4), (1,10), (2,14), (8,10), (9,0)])
#commands = [('y',7), ('x',5)]

def fold(points, command):
    direction = 0 if command[0] == 'x' else 1
    fold_position = command[1]
    new_points = set()
    for point in points:
        if point[direction] == fold_position:
            continue
        if point[direction] > fold_position:
            new_val= 2 * fold_position - point[direction]
        
            if direction:
                new_point = (point[0], new_val)
            else:
                new_point = (new_val, point[1])

            new_points.add(new_point)

        else:
            new_points.add(point)

    return new_points


def fold_everything(points, commands):
    max_x = max_y = 0
    for command in commands:
        points = fold(points, command)
    
    return points


def find_max(points):
    max_x = max_y = 0
    for point in points:
        if point[0] > max_x:
            max_x = point[0]
        if point[1] > max_y:
            max_y = point[1]
    
    return max_x+1, max_y+1


def build_array(points):
    max_x, max_y = find_max(points)
    arr = [[' ' for i in range(max_x)] for j in range(max_y)]

    for point in points:
        x, y = point
        arr[y][x] = '#'

    return arr


def print_array(arr):
    for line in arr:
        print(line)


#print(len(fold(points, commands[0])))
folded_points = fold_everything(points, commands)
array = build_array(folded_points)
print_array(array)



import numpy as np

points = []
with open('input.txt', 'r') as file:
    line = file.readline()
    while line:
        line = line.strip().split(' -> ')
        left = [int(x) for x in line[0].split(',')]
        right = [int(x) for x in line[1].split(',')]
        points.append([left, right])
        line = file.readline()


# Part 1
grid = np.zeros((1000, 1000), int)

for line in points:
    if line[0][0] == line[1][0]:
        x = line[0][0]
        y_start = min(line[0][1], line[1][1])
        y_end = max(line[0][1], line[1][1])
        
        for i in range(y_start, y_end+1):
            grid[x][i] += 1

    elif line[0][1] == line[1][1]:
        y = line[0][1]
        x_start = min(line[0][0], line[1][0])
        x_end = max(line[0][0], line[1][0])

        for i in range(x_start, x_end+1):
            grid[i][y] += 1

    # Part 2
    else:
        max_x = max(line[0][0], line[1][0])
        min_x = min(line[0][0], line[1][0])
        dist = 1 + max_x - min_x
        x_dir = 1 if line[1][0] - line[0][0] > 0 else -1
        y_dir = 1 if line[1][1] - line[0][1] > 0 else -1
        x_start, y_start = line[0][0], line[0][1]

        for i in range(dist):
            grid[x_start + (i*x_dir)][y_start + (i*y_dir)] += 1

count = np.count_nonzero(grid > 1)
print(count)

with open('input.txt', 'r') as file:
    elevation = [line.strip() for line in file.readlines()]
    elevation = [[int(x) for x in line] for line in elevation]

# Part 1
sum = 0
for i, row in enumerate(elevation):
    for j, position in enumerate(row):
        # Up
        if i == 0:
            up = 9
        else:
            up = elevation[i-1][j]
        
        # Down
        if i == len(elevation) - 1:
            down = 9
        else:
            down = elevation[i+1][j]

        # Right
        if j == len(row) - 1:
            right = 9
        else:
            right = elevation[i][j+1]
        
        # Left
        if j == 0:
            left = 9
        else:
            left = elevation[i][j-1]
    
        if position < min(up, down, right, left):
            sum += position + 1

#print(sum)

# Part 2
points = set()
basins = []

def recursive_search(starting_point, sum=0):
    global points
    x, y = starting_point

    if (x, y) in points or elevation[x][y] == 9:
        return sum
    
    else:
        points.add((x, y))
        neighbors = neighbor_search(x, y)
        for neighbor in neighbors:
            i, j, value = neighbor
            if value == 9:
                points.add((i, j))
            elif (i, j) not in points:
                sum = recursive_search((i,j), sum)
    
    return 1 + sum
    

def neighbor_search(x, y):
    # Up
    if x == 0:
        up = [x-1, y, 9]
    else:
        up = [x-1, y, elevation[x-1][y]]
    
    # Down
    if x == len(elevation) - 1:
        down = [x+1, y, 9]
    else:
        down = [x+1, y, elevation[x+1][y]]

    # Right
    if y == len(row) - 1:
        right = [x, y+1, 9]
    else:
        right = [x, y+1, elevation[x][y+1]]
    
    # Left
    if y == 0:
        left = [x, y-1, 9]
    else:
        left = [x, y-1, elevation[x][y-1]]

    return [up, down, left, right]


for i, row in enumerate(elevation):
    for j, position in enumerate(row):
        if (i, j) in points:
            continue

        neighbors = [cell[-1] for cell in neighbor_search(i, j)]

        if position < min(neighbors):
            basin = recursive_search((i, j))
            print(basin)
            if len(basins) < 3:
                basins.append(basin)
            elif basin > min(basins):
                basins.remove(min(basins))
                basins.append(basin)

print(basins)
basin_sum = basins[0] * basins[1] * basins[2]
print(basin_sum)
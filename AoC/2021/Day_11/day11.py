import numpy as np
from numpy.core.numeric import count_nonzero

with open('input.txt', 'r') as file:
    octopi = [line.strip() for line in file.readlines()]
    octopi = [[int(x) for x in line] for line in octopi]
    octopi = np.array(octopi)

STEPS = 100
counter = 0
to_pop = []


def triggered(row, colm):
    valids = []
    width = len(octopi[0])
    height = len(octopi)

    # Top Left
    if row - 1 >= 0 and colm - 1 >= 0:
        valids.append((row-1, colm-1))
    # Top
    if row - 1 >= 0:
        valids.append((row-1, colm))
    # Top Right
    if row - 1 >= 0 and colm + 1 < width:
        valids.append((row-1, colm+1))
    # Left
    if colm - 1 >= 0:
        valids.append((row, colm-1))
    # Right
    if colm + 1 < width:
        valids.append((row, colm+1)) 
    # Bot Left
    if row + 1 < height and colm - 1 >= 0:
        valids.append((row+1, colm-1)) 
    # Bot
    if row + 1 < height:
        valids.append((row+1, colm))
    # Bot Right
    if row + 1 < height and colm + 1 < width:
        valids.append((row+1, colm+1))
    
    for cell in valids:
        row, colm = cell
        if octopi[row][colm] != 0 and (row, colm) not in to_pop:
            octopi[row][colm] += 1
            if octopi[row][colm] > 9:
                to_pop.append((row, colm))
    
for step in range(STEPS):
    octopi += 1
    for row_index, row in enumerate(octopi):
        for octopus_index, octopus in enumerate(row):
            if octopus > 9:
                to_pop.append((row_index, octopus_index))

    while to_pop:
        counter += 1
        row, colm = to_pop.pop()
        octopi[row][colm] = 0
        triggered(row, colm)

print(counter)

# Part 2

with open('input.txt', 'r') as file:
    octopi = [line.strip() for line in file.readlines()]
    octopi = [[int(x) for x in line] for line in octopi]
    octopi = np.array(octopi)

counter = 0
to_pop = []

while True:
    octopi += 1
    for row_index, row in enumerate(octopi):
        for octopus_index, octopus in enumerate(row):
            if octopus > 9:
                to_pop.append((row_index, octopus_index))

    while to_pop:
        row, colm = to_pop.pop()
        octopi[row][colm] = 0
        triggered(row, colm)

    counter += 1

    if np.all(octopi == 0):
        break

print(counter)
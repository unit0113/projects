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

'''octopi = [[5,4,8,3,1,4,3,2,2,3],
[2,7,4,5,8,5,4,7,1,1],
[5,2,6,4,5,5,6,1,7,3],
[6,1,4,1,3,3,6,1,4,6],
[6,3,5,7,3,8,5,4,7,8],
[4,1,6,7,5,2,4,6,4,5],
[2,1,7,6,8,4,1,7,2,1],
[6,8,8,2,8,8,1,1,3,4],
[4,8,4,6,8,4,8,5,5,4],
[5,2,8,3,7,5,1,5,2,6]]
octopi = np.array(octopi)'''


STEPS = 300
to_pop = []

for step in range(STEPS):
    print(step)
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

    if np.all(octopi == 0):
        print(step + 1)
        break





'''
from time import perf_counter as pfc
from scipy.signal import convolve2d
#
grid = np.array([list(i) for i in open('input.txt', 'r').read().strip().splitlines()], dtype='int')
#
def flash_count1(g, lim):
  c = []
  while lim > 0:
    # my first approach was by position finding and recursion...
    # with some help and inspiration.
    # done by convolution of the mask - all neighbouring octopi and the octopi to be flashed in
    # flashing (is Truth array).
    mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    flashed = np.zeros(grid.shape, dtype='bool')
    g += 1
    while np.any(flashing := g > 9):
      flashed |= flashing
      g += convolve2d(flashing, mask, mode='same')
      g[flashed] = 0
    # and finally append the sum of flashed octopi to a list
    c += [flashed.sum()]
    lim -= 1
  return sum(c)
#
def flash_count2(g):
  c = []
  lim = 0
  # just use same function, but counting up to the point....
  while True:
    mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    flashed = np.zeros(grid.shape, dtype='bool')
    g += 1
    while np.any(flashing := g > 9):
      flashed |= flashing
      g += convolve2d(flashing, mask, mode='same')
      g[flashed] = 0
    c += [flashed.sum()]
    lim += 1
    # ...where all octopi are flashing and g has just zeros.
    if np.all(g == 0):
      step = lim
      return step
#
# Part 1:
start1 = pfc()
print('Part 1 result is:', flash_count1(grid.copy(), 100), ', t =', pfc()-start1)
# Part 2:
start2 = pfc()
print('Part 2 result is:', flash_count2(grid), ', t =', pfc()-start2)'''
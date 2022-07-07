import numpy as np
from scipy.signal import convolve2d

octopi = np.array([list(i) for i in open('input.txt', 'r').read().strip().splitlines()], dtype='int')

STEPS = 100
mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

# Part 1
counter = 0
for step in range(STEPS):
    flashed = np.zeros(octopi.shape, dtype='bool')
    octopi += 1
    while np.any(flashing := octopi > 9):
      flashed |= flashing
      octopi += convolve2d(flashing, mask, mode='same')
      octopi[flashed] = 0
    counter += flashed.sum()

#print(counter)

octopi = np.array([list(i) for i in open('input.txt', 'r').read().strip().splitlines()], dtype='int')

# Part 2
counter = 0
while True:
    flashed = np.zeros(octopi.shape, dtype='bool')
    octopi += 1
    while np.any(flashing := octopi > 9):
      flashed |= flashing
      octopi += convolve2d(flashing, mask, mode='same')
      octopi[flashed] = 0
    counter += 1

    # ...where all octopi are flashing and g has just zeros.
    if np.all(octopi == 0):
      break

print(counter)
from random import random


num_half_lives = int(input('Enter number of half lives: '))
num_particles = 100

for half_life in range(1, num_half_lives + 1):
    remaining = 0
    for _ in range(num_particles):
        if random() >= 0.5:
            remaining += 1
    print(f'Remaining after half life {half_life}: {remaining}')
    num_particles = remaining

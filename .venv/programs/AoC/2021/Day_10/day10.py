import numpy as np

with open('input.txt', 'r') as file:
    input = [line.strip() for line in file.readlines()]
    input = [[x for x in line] for line in input]

# Part 1
openers = ['(', '[', '{', '<']
closers = [')', ']', '}', '>']
points = [3, 57, 1197, 25137]
counters = [0] * 4

for line in input:
    corrupted = False
    for index, char in enumerate(line):
        if corrupted:
            break
        if char in closers:
            for line_index in range(index - 1, 0, -1):
                if line[line_index] in openers and line[line_index] != openers[closers.index(char)]:
                    counters[closers.index(char)] += 1
                    corrupted = True
                    break
                elif line[line_index] == openers[closers.index(char)]:
                    line[line_index] = line[index] = '*'
                    break

#print(sum(np.multiply(points, counters)))

# Part 2
openers = ['(', '[', '{', '<']
closers = [')', ']', '}', '>']
points = [1, 2, 3, 4]
counters = [0] * 4
to_delete = []

for del_index, line in enumerate(input):
    corrupted = False
    for index, char in enumerate(line):
        if corrupted:
            break
        if char in closers:
            for line_index in range(index - 1, -1, -1):
                if line[line_index] in openers and line[line_index] != openers[closers.index(char)]:
                    corrupted = True
                    break
                elif line[line_index] == openers[closers.index(char)]:
                    line[line_index] = line[index] = '*'
                    break
    if corrupted:
        to_delete.append(del_index)

to_delete.reverse()
for index in to_delete:
    del input[index]

finishers = []
for line in input:
    line_finisher = []
    line.reverse()
    for char in line:
        if char in openers:
            line_finisher.append(closers[openers.index(char)])
    finishers.append(line_finisher)

answer = []
for line in finishers:
    total = 0
    for char in line:
        total *= 5
        total += points[closers.index(char)]
    answer.append(total)

answer.sort()
print(answer[(len(answer) -1) // 2])
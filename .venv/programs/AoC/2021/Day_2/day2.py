with open('input.txt', 'r') as file:
    direction_list = [line.split() for line in file]

# Part 1
horizontal = depth = 0
for line in direction_list:
    if line[0] == 'forward':
        horizontal += int(line[1])
    elif line[0] == 'up':
        depth -= int(line[1])
    elif line[0] == 'down':
        depth += int(line[1])

answer = horizontal * depth
print(answer)

# Part 2
horizontal = depth = aim = 0
for line in direction_list:
    if line[0] == 'forward':
        horizontal += int(line[1])
        depth += aim * int(line[1])
    elif line[0] == 'up':
        aim -= int(line[1])
    elif line[0] == 'down':
        aim += int(line[1])

answer = horizontal * depth
print(answer)
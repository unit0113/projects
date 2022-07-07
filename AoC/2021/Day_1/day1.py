with open('input.txt', 'r') as file:
    depth_list = [int(line) for line in file]

# Part 1
increase_count = 0
for i, depth in enumerate(depth_list[1:]):
    if depth > depth_list[i]:
        increase_count += 1

print(increase_count)

# Part 2
increase_count = 0
for i in range(len(depth_list) -3):
    if sum(depth_list[i+1:i+4]) > sum(depth_list[i:i+3]):
        increase_count += 1

print(increase_count)
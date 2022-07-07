groups = [0] * 9

with open('input.txt', 'r') as file:
    line = file.readline().split(',')
    for fish in line:
        groups[int(fish)] += 1

DAYS = 256

for day in range(DAYS):
    new_groups = [0] * 9
    for age, size in enumerate(groups):
        if age == 0:
            new_groups[8] = new_groups[6] = size
        else:
            new_groups[age-1] += size

    groups = new_groups.copy()

answer = sum(groups)
print(answer)
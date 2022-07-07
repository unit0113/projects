with open('input.txt', 'r') as file:
    crabs = [int(crab) for crab in file.readline().split(',')]

# Part 1
fuel_min = 1000000000000000000000000000000000000000000000000000000000000
min_position = 0
for position in range(min(crabs), max(crabs)):
    fuel_cost = 0
    for crab in crabs:
        fuel_cost += abs(crab - position)
    if fuel_cost < fuel_min:
        fuel_min = fuel_cost
        min_position = position

#print(min_position, fuel_min)

# Part 2
fuel_min = 1000000000000000000000000000000000000000000000000000000000000
min_position = 0
fuel_cost_list = [0]
for index in range(min(crabs), max(crabs) + 1)[1:]:
    fuel_cost_list.append(index + fuel_cost_list[-1])

for position in range(min(crabs), max(crabs)):
    fuel_cost = 0
    for crab in crabs:
        dist = abs(crab - position)
        fuel_cost += fuel_cost_list[dist]
        
    if fuel_cost < fuel_min:
        fuel_min = fuel_cost
        min_position = position

print(min_position, fuel_min)
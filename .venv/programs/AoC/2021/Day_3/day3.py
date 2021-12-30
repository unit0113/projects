with open('input.txt', 'r') as file:
    input_list = [line.strip() for line in file]

# Part 1
gamma = epsilon = ''
for i in range(len(input_list[0])):
    index_list = [int(code[i]) for code in input_list]
    index_sum = sum(index_list)
    if index_sum >= 500:
        gamma += '1'
        epsilon += '0'
    else:
        gamma += '0'
        epsilon += '1'

gamma = int(gamma, 2)
epsilon = int(epsilon, 2)

answer = gamma * epsilon
#print(answer)

# Part 2
edited_list = input_list.copy()
for i in range(len(input_list[0])):
    index_list = [int(code[i]) for code in edited_list]
    index_sum = sum(index_list)
    if index_sum >= len(edited_list) / 2:
        value = '1'
    else:
        value = '0'

    edited_list = [code for code in edited_list if code[i] == value]
    if len(edited_list) == 1:
        break

oxygen = int(edited_list[0], 2)

edited_list = input_list.copy()
for i in range(len(input_list[0])):
    index_list = [int(code[i]) for code in edited_list]
    index_sum = sum(index_list)
    if index_sum >= len(edited_list) / 2:
        value = '0'
    else:
        value = '1'

    edited_list = [code for code in edited_list if code[i] == value]
    if len(edited_list) == 1:
        break

carbon = int(edited_list[0], 2)
answer = oxygen * carbon
print(answer)
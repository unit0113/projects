with open('input.txt', 'r') as file:
    lines = [line.split('|') for line in file.readlines()]
    
    inputs = []
    outputs = []
    for line in lines:
        inputs.append(line[0].split())
        outputs.append(line[1].split())

# Part 1
count = 0
for output in outputs:
    output_len = [len(out) for out in output]
    count += sum([1 for item in output_len if item in [2, 4, 3, 7]])

#print(count)

# Part 2
sum = 0
for input, output in zip(inputs, outputs):
    input_len = [len(code) for code in input]
    sorted_input = []
    for code in input:
        sorted_input.append(set([ch for ch in code]))
    sorted_output = [''.join(sorted(code)) for code in output]

    one = sorted_input[input_len.index(2)]
    four = sorted_input[input_len.index(4)]
    seven = sorted_input[input_len.index(3)]
    eight = sorted_input[input_len.index(7)]

    for option in [code for code in sorted_input if len(code) == 6]:
        if one <= option and not four <= option:
            zero = option
            break

    for option in [code for code in sorted_input if len(code) == 6]:
        if zero != option and four <= option:
            nine = option
            break
    
    for option in [code for code in sorted_input if len(code) == 6]:
        if zero != option and nine != option:
            six = option
            break
    
    for option in [code for code in sorted_input if len(code) == 5]:
        if one <= option:
            three = option
            break
    
    bl = eight.copy()
    bl.difference_update(nine)
    for option in [code for code in sorted_input if len(code) == 5]:
        if option != three:
            if bl <= option:
                two = option
            else:
                five = option

    decoded_input = [''.join(sorted(zero)), ''.join(sorted(one)), ''.join(sorted(two)), ''.join(sorted(three)), ''.join(sorted(four)), ''.join(sorted(five)), ''.join(sorted(six)), ''.join(sorted(seven)), ''.join(sorted(eight)), ''.join(sorted(nine))]
    answer = ''
    for output_code in sorted_output:
        answer += str(decoded_input.index(output_code))

    sum += int(answer)
        
print(sum)